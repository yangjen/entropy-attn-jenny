# Decode Phase Investigation - Bug Report

**Branch**: `m-feedback`
**Date**: 2026-02-15
**Investigation Focus**: Decode phase feedback loop and controller behavior

---

## Critical Issues Found

### BUG #1: Controller State Leaking Between Examples 🚨

**Severity**: CRITICAL - Invalidates all evaluation results

**Evidence**:
Examining `logs/qa_1_entropy_attn_entropy_logs_scaled.jsonl`:
```
Example 0: temp_mean=1.0000 (all steps)
Example 1: temp_mean=1.0000 → 0.9997 → 0.9994
Example 2: temp_mean=0.9989 (starts here, should be 1.0!)
Example 3: temp_mean=0.9941 (starts here)
Example 4: temp_mean=0.9912 (starts here)
...
Example 49: temp_mean=0.8928 (starts here)
```

Temperature drifts from 1.0 → 0.89 across examples. State is not reset.

**Root Cause**:
- `models/attn_patch.py:69-70`: Controller initialized once per layer
- `run_ruler_eval_timed.py:297-327`: `reset_entropy_controller()` only resets `prompt_target_entropy`, not temperature or EMA state

**Expected Behavior**:
Each example should start with `temp=1.0`, `ema_entropy=0`, `prompt_target_entropy=None`

**Impact**:
- Later examples run with artificially lowered temperature
- Baseline vs scaled comparisons are invalid
- Measured improvements/degradations are artifacts of state leakage

**Files Affected**:
- `models/attn_patch.py:69-84`
- `run_ruler_eval_timed.py:297-327`
- `models/entropy_scaling.py:46-54` (needs reset method)

---

### BUG #2: Controller Over-Aggressive & Saturating ⚠️

**Severity**: HIGH - Controller ineffective due to constant boundary saturation

**Evidence**:
From `logs/entropy_scaling_analysis.ipynb` controller health metrics:
```
temp_sat_frac: 0.4864    # Hits bounds 48.6% of time
oscillation:   0.4880    # Changes direction 48.8% of steps
temp_range:    0.0056    # Only 0.5% dynamic range used
```

**Analysis**:
- Controller hits `temp_min` or `temp_max` nearly half the time
- Temperature oscillates wildly (changes direction every other step)
- Effective temperature range: [~0.88, ~1.0] instead of designed [0.7, 1.0]
- Controller spends most time clamped at boundaries

**Root Cause**:
Current parameters cause excessive gain:
```python
kp = 0.35           # Proportional gain (too high)
max_step = 0.0005   # Step limit (actual value from run_ruler_eval_timed.py)
ema_beta = 0.9      # EMA smoothing (not enough)
temp_min = 0.7      # Lower bound
temp_max = 1.0      # Upper bound
```

Even small errors: `0.35 * err > 0.0005` → saturate immediately

**Expected Behavior**:
Ideal controller metrics:
- `temp_sat_frac < 0.15` (< 15% time at bounds)
- `oscillation < 0.25` (< 25% direction changes)
- `temp_range ~ 0.15-0.20` (using 50%+ of available range)

**Impact**:
- Controller mostly operates in saturated "bang-bang" mode
- Unable to make fine-grained adjustments
- Reacts to noise rather than signal
- Feedback loop becomes ineffective

**Files Affected**:
- `models/attn_patch.py:70-78` (controller initialization)
- `models/entropy_scaling.py:23-36` (default parameters)

---

### BUG #3: Inverted Dose-Response Relationship 🔥

**Severity**: HIGH - Controller having opposite effect from intended

**Evidence**:
From `logs/entropy_scaling_analysis.ipynb` dose-response analysis:
- Larger `|ΔT|` (temperature changes) → **HIGHER** entropy next step
- Expected: Larger `|ΔT|` → **LOWER** entropy (sharper attention)

**Analysis**:
Controller appears to be reacting to noise rather than causal signal:

1. **Natural entropy spike occurs** (e.g., token forces broad attention)
2. **Controller reacts**: Reduces temperature to sharpen
3. **By next step**: Natural spike already gone
4. **Result**: Lower temperature prevents entropy from staying naturally low
5. **Net effect**: Temperature changes correlate with entropy increases

This is a classic **phase lag problem** in control systems.

**Hypothesis**:
The entropy "spikes" the controller sees are not persistent patterns to correct, but transient noise. The controller's reaction arrives too late and actually amplifies variance.

**Contributing Factors**:
- EMA smoothing (`beta=0.9`) may not be enough
- No dead-zone: Controller reacts to every small fluctuation
- Single-token decode makes it impossible to "look ahead"
- Controller optimizing per-token entropy, not sequence-level quality

**Impact**:
- Controller may be **degrading** attention quality
- Explains why improvements over baseline are minimal/negative
- Temperature modulation correlates with worse behavior

**Files Affected**:
- `models/entropy_scaling.py:59-105` (update logic)
- `models/attn_patch.py:96-117` (target setting)
- Control algorithm design (may need fundamental rethink)

---

### BUG #4: Triton vs PyTorch Entropy Inconsistency (Unverified) ⚠️

**Severity**: MEDIUM - Risk of mismatched entropy measurements

**Evidence**:
Two different entropy computation paths:

**Triton (prefill)** - `models/entropy_attn_triton.py:240`:
```python
e_i = (tl.math.log2(l_i) + m_i - (e_i / l_i)) * ln2
```
Uses log-sum-exp with base-2 logs for numerical stability.

**PyTorch (decode)** - `models/entropy_attn_triton.py:563`:
```python
entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=-1)
```
Uses natural log directly.

**Risk**:
If numerical differences exist:
- Prompt target entropy (from Triton) won't match decode entropy (from PyTorch)
- Controller chases a mismatched target
- Systematic bias in error signal

**Status**: UNVERIFIED - No test exists to check this

**Files Affected**:
- `models/entropy_attn_triton.py:240-246` (Triton entropy)
- `models/entropy_attn_triton.py:563` (PyTorch entropy)

---

## Secondary Issues

### Issue #5: No Per-Sample Entropy Logging for Debugging

**Observation**:
Entropy logs only capture mean/std across heads, not per-head or per-layer details.

**Impact**:
- Cannot debug head-specific behavior
- Cannot see if certain heads are causing oscillation
- Limited visibility into GQA group behavior

**Files Affected**:
- `models/attn_patch.py:130-144` (logging logic)

---

### Issue #6: GQA Temperature Sharing Not Considered

**Observation**:
Llama uses Grouped Query Attention (32 Q heads, 8 KV heads). Each Q head gets independent temperature, but heads sharing KV might need coordinated temps.

**Analysis**:
Currently: 32 independent temperatures per layer
Question: Should heads sharing same KV group share temperature?

**Impact**: Unknown - requires testing

**Files Affected**:
- `models/attn_patch.py:37-38` (repeat_kv)
- `models/entropy_scaling.py` (per-head state)

---

## Recommended Fixes (Priority Order)

### Priority 1: Fix State Leakage (BUG #1)

**Change `run_ruler_eval_timed.py:reset_entropy_controller()`**:
```python
def reset_entropy_controller(model):
    for module in model.modules():
        if hasattr(module, "_entropy_temp_controller"):
            controller = module._entropy_temp_controller
            # Reset ALL state, not just target
            controller.temp = None
            controller.ema_entropy = None
            controller.prompt_target_entropy = None
```

**Alternative**: Add explicit reset method to `EntropyTempController`

---

### Priority 2: Reduce Controller Aggression (BUG #2)

**Option A - Reduce Gain**:
```python
# In models/attn_patch.py:76-77
kp=0.15,  # Down from 0.35
ema_beta=0.95,  # Up from 0.9
```

**Option B - Increase Smoothing**:
```python
ema_beta=0.98,  # Much stronger smoothing
kp=0.35,  # Keep current
```

**Option C - Add Dead-Zone**:
```python
# In models/entropy_scaling.py:90-91
err = self.ema_entropy - target
err = err.clamp(min=0.0)
# Add:
err = torch.where(torch.abs(err) < 0.05, torch.zeros_like(err), err)
```

---

### Priority 3: Verify Entropy Consistency (BUG #4)

Create unit test comparing Triton and PyTorch entropy on identical inputs.

---

### Priority 4: Add Comprehensive Logging

Log per-head entropy and temperature for detailed analysis.

---

## Unit Tests Needed

1. **Test: Controller state reset between examples**
   - Verify temp, ema_entropy, prompt_target all reset

2. **Test: Triton vs PyTorch entropy equivalence**
   - Same Q, K, V, temp → same entropy (within tolerance)

3. **Test: Controller doesn't saturate excessively**
   - Run on synthetic data, check `temp_sat_frac < 0.2`

4. **Test: Controller stability (no oscillation)**
   - Check sign changes in consecutive steps

5. **Test: Dose-response relationship**
   - Lower temp → lower entropy (on controlled test case)

---

## Questions for Further Investigation

1. **Why is dose-response inverted?**
   - Is this a fundamental issue with per-token feedback?
   - Should we use multi-token lookahead?

2. **Should we optimize for per-token entropy or sequence-level metric?**
   - Current: Per-token entropy minimization
   - Alternative: Sequence perplexity, task accuracy

3. **Is the prompt target entropy actually meaningful?**
   - Tail-256 trimmed mean: Is this the right reference?
   - Should we adapt target during decode?

4. **Does temperature scaling help at all?**
   - After fixing bugs, do we see any benefit?
   - Maybe static temp is better?

---

## Files to Modify

### Core Implementation:
- `models/entropy_scaling.py` - Add reset(), tune params
- `models/attn_patch.py` - Fix initialization, add better logging
- `run_ruler_eval_timed.py` - Fix reset_entropy_controller()

### Testing:
- `test_kernel.py` - Add entropy consistency test
- `test_controller.py` (NEW) - Unit tests for controller
- `test_integration.py` (NEW) - End-to-end reset verification

### Analysis:
- `logs/entropy_scaling_analysis.ipynb` - Add more diagnostics

---

## Next Steps

1. Implement unit tests to reproduce BUG #1 and BUG #2
2. Verify BUG #4 (entropy consistency)
3. Create patches for Priority 1 and 2 fixes
4. Re-run evaluation with fixes
5. Analyze if improvements appear after bug fixes
