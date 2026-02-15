# Research Logic Analysis & Refactoring Plan

**Goal**: Identify core research logic and refactor for testability

---

## Research Hypothesis

**Core Claim**:
> "Maintaining decode-time attention entropy close to prompt-derived entropy improves generation quality by preventing attention collapse/diffusion."

**Mechanism**:
1. Measure attention entropy during prompt processing (prefill)
2. Extract target entropy from prompt tail (reference point)
3. During decode, measure per-token attention entropy
4. Adjust temperature via feedback control to match target
5. Temperature modulates softmax sharpness → controls entropy

---

## Current Code Architecture

### Component Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Pipeline                         │
└─────────────────────────────────────────────────────────────┘
         │
         ├─► PREFILL (N_CTX > 1)
         │   ├─ Triton kernel: Compute attention + entropy
         │   │  └─ entropy_attn_triton.py:240-246
         │   ├─ Normalize entropy: H / log(kv_len)
         │   │  └─ attn_patch.py:99-101
         │   ├─ Extract tail (last K tokens)
         │   │  └─ attn_patch.py:104-105
         │   ├─ Compute trimmed mean as target
         │   │  └─ attn_patch.py:108-116
         │   └─ Set target: controller.set_prompt_target()
         │      └─ entropy_scaling.py:50-54
         │
         └─► DECODE (N_CTX == 1)
             ├─ PyTorch: Compute attention + entropy
             │  └─ entropy_attn_triton.py:555-563
             ├─ Normalize entropy
             │  └─ entropy_scaling.py:73-77
             ├─ EMA smoothing
             │  └─ entropy_scaling.py:82-83
             ├─ Compute error: H_ema - H_target
             │  └─ entropy_scaling.py:86-91
             ├─ Proportional control: delta = -kp * err
             │  └─ entropy_scaling.py:98-100
             ├─ Update temperature: T += delta
             │  └─ entropy_scaling.py:102-103
             └─ Apply temp in next forward pass
                └─ entropy_attn_triton.py:561 (PyTorch)
                └─ attn_patch.py:86 (passed to kernel)
```

---

## Code That Implements Research Logic

### 1. **Entropy Computation** (TANGLED ⚠️)

**Location**:
- Triton kernel: `entropy_attn_triton.py:240-246`
- PyTorch fallback: `entropy_attn_triton.py:563`

**Current State**:
```python
# Triton (inside kernel, fused with attention)
e_i = (tl.math.log2(l_i) + m_i - (e_i / l_i)) * ln2

# PyTorch (standalone)
entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
```

**Problem**:
- Entropy calc is fused into attention kernel
- Can't test entropy independently
- Two different implementations (Triton vs PyTorch)

**Refactoring Needed**:
Extract into pure function:
```python
def compute_entropy_from_attention_weights(attn_weights: Tensor) -> Tensor:
    """
    Compute Shannon entropy: H = -sum(p * log(p))

    Args:
        attn_weights: [*, seq_len] - Normalized attention weights (sum to 1)
    Returns:
        entropy: [*] - Entropy per attention distribution
    """
    return -(attn_weights * torch.log(attn_weights + 1e-9)).sum(dim=-1)
```

**Testability**: Can feed synthetic attention distributions (peaked, flat, uniform)

---

### 2. **Entropy Normalization** (EASY TO EXTRACT ✓)

**Location**:
- `attn_patch.py:99-101` (prefill)
- `entropy_scaling.py:73-77` (decode)

**Current State**:
```python
# attn_patch.py
H_norm = attn_entropy / torch.log(
    torch.tensor(float(kv_len), device=attn_entropy.device)
).clamp(min=1.0)

# entropy_scaling.py (nearly identical)
norm = torch.log(
    torch.tensor(float(kv_len), device=entropy_last.device)
).clamp(min=1.0)
H_norm = entropy_last / norm
```

**Problem**: Duplicated logic, device handling mixed with logic

**Refactoring**:
```python
def normalize_entropy(entropy: Tensor, kv_len: int) -> Tensor:
    """
    Normalize entropy by log(sequence_length) to make comparable across lengths.

    Args:
        entropy: Raw entropy values
        kv_len: KV cache length (sequence length)
    Returns:
        Normalized entropy in [0, 1] range
    """
    norm = torch.log(torch.tensor(float(kv_len), device=entropy.device))
    norm = norm.clamp(min=1.0)  # Avoid division by zero
    return entropy / norm
```

**Testability**: Test with kv_len=1, 10, 100, 1000. Verify output range [0, 1].

---

### 3. **Target Entropy from Tail** (MODERATELY TANGLED ⚠️)

**Location**: `attn_patch.py:96-117`

**Current State**:
```python
if N_CTX > 1 and controller.prompt_target_entropy is None:
    kv_len = key.shape[2]

    H_norm = attn_entropy / torch.log(
        torch.tensor(float(kv_len), device=attn_entropy.device)
    ).clamp(min=1.0)

    K = min(256, H_norm.shape[-1])
    tail = H_norm[:, :, -K:]

    trim_ratio = float(getattr(module, "target_trim_ratio", 0.0))
    trim_ratio = max(0.0, min(0.49, trim_ratio))
    trim_n = int(K * trim_ratio)

    if trim_n > 0 and (2 * trim_n) < K:
        tail_sorted = torch.sort(tail, dim=-1).values
        tail_core = tail_sorted[:, :, trim_n:(K - trim_n)]
        prompt_target = tail_core.mean(dim=-1, keepdim=True)
    else:
        prompt_target = tail.mean(dim=-1, keepdim=True)

    controller.set_prompt_target(prompt_target)
```

**Problem**:
- Mixed with control flow (if N_CTX > 1)
- Module attribute access (getattr)
- Hard to test in isolation

**Refactoring**:
```python
def compute_target_from_tail(
    entropy_normalized: Tensor,  # [Z, H, seq_len]
    tail_length: int = 256,
    trim_ratio: float = 0.0
) -> Tensor:
    """
    Compute target entropy from tail of sequence using trimmed mean.

    Args:
        entropy_normalized: Normalized entropy sequence [Z, H, seq_len]
        tail_length: Number of tokens to use from end (default: 256)
        trim_ratio: Fraction to trim from each end [0, 0.49] (default: 0)
    Returns:
        target_entropy: [Z, H, 1] - Target for decode phase

    Example:
        >>> H = torch.randn(1, 32, 1000)  # 1000 token prompt
        >>> target = compute_target_from_tail(H, tail_length=256, trim_ratio=0.1)
        >>> target.shape
        torch.Size([1, 32, 1])
    """
    # Clamp inputs
    trim_ratio = max(0.0, min(0.49, trim_ratio))
    K = min(tail_length, entropy_normalized.shape[-1])

    # Extract tail
    tail = entropy_normalized[:, :, -K:]

    # Trimmed mean
    trim_n = int(K * trim_ratio)
    if trim_n > 0 and (2 * trim_n) < K:
        tail_sorted = torch.sort(tail, dim=-1).values
        tail_core = tail_sorted[:, :, trim_n:(K - trim_n)]
        return tail_core.mean(dim=-1, keepdim=True)
    else:
        return tail.mean(dim=-1, keepdim=True)
```

**Testability**:
- Synthetic sequences with known statistics
- Test trim_ratio = 0.0, 0.1, 0.2
- Test with outliers (should be trimmed)
- Edge cases: short sequences (K < tail_length)

---

### 4. **EMA Update** (EASY TO EXTRACT ✓)

**Location**: `entropy_scaling.py:82-83`

**Current State**:
```python
ema_new = self.ema_entropy * self.ema_beta + H_safe * (1 - self.ema_beta)
self.ema_entropy = torch.where(valid_entropy, ema_new, self.ema_entropy)
```

**Problem**: Stateful, mixed with validity checking

**Refactoring**:
```python
def update_ema(
    ema_prev: Tensor,
    value_new: Tensor,
    beta: float,
    valid_mask: Optional[Tensor] = None
) -> Tensor:
    """
    Exponential moving average update.

    Args:
        ema_prev: Previous EMA value
        value_new: New observation
        beta: Smoothing factor [0, 1]. Higher = more smoothing
        valid_mask: Optional mask for which values to update
    Returns:
        ema_updated: New EMA value

    Formula: EMA_new = beta * EMA_old + (1 - beta) * value_new
    """
    ema_new = ema_prev * beta + value_new * (1 - beta)

    if valid_mask is not None:
        return torch.where(valid_mask, ema_new, ema_prev)
    return ema_new
```

**Testability**:
- Test with beta=0, 0.5, 0.9, 0.99
- Verify convergence properties
- Test with step input (measure settling time)

---

### 5. **Control Delta Computation** (CORE RESEARCH LOGIC ⭐)

**Location**: `entropy_scaling.py:86-100`

**Current State**:
```python
if self.prompt_target_entropy is not None:
    valid_target = torch.isfinite(self.prompt_target_entropy)
    valid = valid_entropy & valid_target
    target = torch.where(valid_target, self.prompt_target_entropy, torch.zeros_like(...))
    err = self.ema_entropy - target
    err = torch.clamp(err, min=0.0)  # Only correct upward entropy
else:
    valid = valid_entropy
    err = self.ema_entropy  # Fallback: pure sharpening

delta = -self.kp * err
delta = delta.clamp(-self.max_step, self.max_step)
delta = torch.where(valid, delta, torch.zeros_like(delta))
```

**Problem**:
- Stateful (self.ema_entropy, self.prompt_target_entropy)
- Mixed validity logic
- **CRITICAL RESEARCH DECISION**: `err = torch.clamp(err, min=0.0)` - Only reduces temp, never increases!

**Refactoring**:
```python
def compute_temperature_delta(
    entropy_current: Tensor,  # [Z, H, 1]
    entropy_target: Tensor,   # [Z, H, 1]
    kp: float,
    max_step: float,
    allow_increase: bool = False
) -> Tensor:
    """
    Compute temperature adjustment using proportional control.

    Args:
        entropy_current: Current (EMA-smoothed) entropy
        entropy_target: Target entropy from prompt
        kp: Proportional gain (typical: 0.1-0.5)
        max_step: Maximum absolute temperature change per step
        allow_increase: If False, only decrease temperature (sharpen only)
    Returns:
        delta_temp: Temperature change [-max_step, +max_step]

    Control Law:
        error = H_current - H_target
        delta_T = -kp * error     (negative feedback)

    If H_current > H_target: error > 0 → delta_T < 0 → decrease temp → sharpen
    If H_current < H_target: error < 0 → delta_T > 0 → increase temp → relax
    """
    err = entropy_current - entropy_target

    if not allow_increase:
        err = torch.clamp(err, min=0.0)  # Only correct high entropy

    delta = -kp * err
    delta = delta.clamp(-max_step, max_step)

    return delta
```

**Testability**: **THIS IS THE KEY FUNCTION TO TEST EXTENSIVELY**
- Test with synthetic error signals
- Verify: err > 0 → delta < 0 (correct direction)
- Test saturation at max_step
- Test allow_increase flag
- **Test dose-response**: Various |delta| → measure effect

---

### 6. **Temperature Update** (TRIVIAL ✓)

**Location**: `entropy_scaling.py:102-103`

**Current State**:
```python
self.temp.add_(delta)
self.temp.clamp_(self.temp_min, self.temp_max)
```

**Refactoring**:
```python
def update_temperature(
    temp_current: Tensor,
    delta: Tensor,
    temp_min: float = 0.7,
    temp_max: float = 1.0
) -> Tensor:
    """
    Update temperature with delta and clamp to valid range.

    Args:
        temp_current: Current temperature
        delta: Change in temperature
        temp_min: Lower bound
        temp_max: Upper bound
    Returns:
        temp_new: Updated and clamped temperature
    """
    temp_new = temp_current + delta
    return temp_new.clamp(temp_min, temp_max)
```

**Testability**: Test boundary cases, verify clamping

---

## Refactoring Priority

### Phase 1: Extract Pure Functions (No Breaking Changes)
1. ✅ `normalize_entropy()` - Used in 2 places
2. ✅ `update_ema()` - Core algorithm
3. ✅ `compute_temperature_delta()` - **CORE RESEARCH LOGIC**
4. ✅ `update_temperature()` - Trivial but good to isolate
5. ✅ `compute_target_from_tail()` - Complex logic worth testing

### Phase 2: Create Test Utilities
6. `generate_synthetic_entropy_sequence()` - For testing
7. `generate_synthetic_attention_weights()` - For entropy testing
8. `mock_controller_update_loop()` - For integration tests

### Phase 3: Refactor Stateful Components
9. Add `EntropyTempController.reset()` method
10. Extract `_init_state()` logic
11. Make controller fully testable in isolation

---

## New File Structure (Proposed)

```
models/
├── entropy_scaling.py           # Stateful controller (as-is for now)
├── entropy_ops.py               # NEW: Pure functions extracted
│   ├── normalize_entropy()
│   ├── compute_entropy_from_attention_weights()
│   ├── compute_target_from_tail()
│   ├── update_ema()
│   ├── compute_temperature_delta()
│   └── update_temperature()
├── attn_patch.py                # Modified to use entropy_ops
└── entropy_attn_triton.py       # As-is (harder to refactor)

tests/
├── test_entropy_ops.py          # NEW: Test pure functions
├── test_controller_unit.py      # Test controller with mocks
└── test_integration.py          # End-to-end scenarios
```

---

## Testing Strategy Per Function

### High Priority (Core Research Logic)

**`compute_temperature_delta()`** - **MOST IMPORTANT**
- [ ] Test: error > 0 → delta < 0 (sharpening)
- [ ] Test: error < 0 → delta > 0 (relaxing) [if allow_increase=True]
- [ ] Test: Clamping at max_step
- [ ] Test: allow_increase=False only sharpens
- [ ] Measure: Gain mapping (|err| → |delta|)
- [ ] Test: With realistic error distributions

**`compute_target_from_tail()`**
- [ ] Test: Trimmed mean vs regular mean
- [ ] Test: With outliers (verify trimming works)
- [ ] Test: Edge case: K < tail_length
- [ ] Test: trim_ratio clamping [0, 0.49]
- [ ] Verify: Output range is plausible

**`normalize_entropy()`**
- [ ] Test: Various sequence lengths
- [ ] Test: Edge case: kv_len=1
- [ ] Verify: Output range [0, 1] for realistic inputs

### Medium Priority (Supporting Logic)

**`update_ema()`**
- [ ] Test: Convergence properties
- [ ] Test: Step response (measure settling time)
- [ ] Test: beta=0, 0.5, 0.9, 0.99
- [ ] Test: valid_mask functionality

**`update_temperature()`**
- [ ] Test: Boundary clamping
- [ ] Test: No clamping when in range

### Lower Priority (Infrastructure)

**`compute_entropy_from_attention_weights()`**
- [ ] Test: Uniform distribution → max entropy
- [ ] Test: One-hot distribution → min entropy
- [ ] Test: Against reference implementation
- [ ] Test: Numerical stability (very small probabilities)

---

## Expected Insights From Testing

1. **Dose-Response Mystery**:
   - Test `compute_temperature_delta()` with controlled inputs
   - If direction is correct in isolation, bug is in system dynamics (timing/phase lag)
   - If direction is wrong, bug is in control law itself

2. **Saturation Issue**:
   - Test with realistic error distributions from logs
   - Measure: What kp/max_step keeps sat_frac < 0.2?
   - Find optimal parameters empirically

3. **Target Selection**:
   - Test with various trim_ratios
   - Question: Is tail-256 trimmed mean actually meaningful?
   - Could test alternative targets (median, percentile, etc.)

---

## Action Plan

1. **Create `models/entropy_ops.py`** with extracted functions
2. **Write tests in `tests/test_entropy_ops.py`**
3. **Refactor `entropy_scaling.py` and `attn_patch.py`** to use new functions
4. **Run tests** with synthetic data to validate behavior
5. **Analyze results** to understand bugs better
6. **Iterate** on parameters/design based on insights

---

## Open Questions

1. **Why `err.clamp(min=0.0)`?**
   - Only sharpens, never relaxes
   - Is this intentional? What's the justification?
   - Should we test with bidirectional control?

2. **Is per-token feedback too fast?**
   - Controller reacts to single-token entropy
   - Should we use multi-token window?

3. **What's the "right" target?**
   - Tail-256 trimmed mean: principled or arbitrary?
   - Could test alternatives

4. **Should temperature be per-head or shared?**
   - Current: 32 independent temps per layer
   - In GQA: heads share KV, should they share temp?

This is more tractable now - we can test the core logic without needing the full model!
