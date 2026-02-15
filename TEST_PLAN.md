# Test Plan for Bug Reproduction & Fixes

**Branch**: `m-feedback`
**Related**: See `BUGS_FOUND.md` for detailed bug descriptions

---

## Testing Philosophy

For each bug, we want:
1. **Reproduction test** - Demonstrates the bug exists
2. **Verification test** - Confirms the bug is fixed after patch
3. **Regression test** - Ensures fix doesn't break other functionality

---

## BUG #1: Controller State Leakage

### Test Strategy

**Goal**: Prove that controller state persists across examples when it shouldn't.

**Approach**:
- Create mock "evaluation loop" that processes 2+ examples sequentially
- Track temperature values at start of each example
- Expected: Each example starts at `temp=1.0`
- Actual (buggy): Example N starts with temp from Example N-1

**Test Structure**:
```python
def test_state_leakage_reproduction():
    """Reproduce BUG #1: State carries over between examples."""
    # Setup: Create controller + mock attention module
    # Example 1: Run prefill + decode, drive temp down
    # Example 2: Start WITHOUT reset
    # Assert: temp != 1.0 at start of Example 2

def test_state_reset_fix():
    """Verify that reset() fixes the leakage."""
    # Same setup as above
    # Example 1: Run, drive temp down
    # Call: controller.reset()  # The fix!
    # Example 2: Start after reset
    # Assert: temp == 1.0 (or None) at start
```

**Files to Test**:
- `models/entropy_scaling.py` - Controller state management
- `run_ruler_eval_timed.py` - reset_entropy_controller() function

**Complexity**: LOW - Pure unit test, no GPU needed

---

## BUG #2: Controller Saturation & Oscillation

### Test Strategy

**Goal**: Measure controller health metrics and show they're outside acceptable range.

**Approach**:
- Run controller with current default params on synthetic entropy sequence
- Calculate metrics: temp_sat_frac, oscillation, temp_range
- Expected: temp_sat_frac < 0.15, oscillation < 0.25
- Actual (buggy): temp_sat_frac > 0.4, oscillation > 0.4

**Test Structure**:
```python
def test_controller_saturation_current_params():
    """Measure saturation with current default parameters."""
    # Create controller with: kp=0.35, ema_beta=0.9, max_step=0.0005
    # Generate synthetic entropy: mean=2.5, std=1.0, length=100
    # Run update() loop, collect temperatures
    # Calculate: sat_frac, oscillation, range
    # Assert: sat_frac > 0.3 (proves bug exists)

def test_controller_improved_params():
    """Show reduced gain improves metrics."""
    # Create controller with: kp=0.15, ema_beta=0.95
    # Same synthetic input
    # Calculate metrics
    # Assert: sat_frac < 0.25 (improvement)

def test_controller_stability_on_step_response():
    """Test controller response to step change in target."""
    # Entropy constant at 2.0, then jumps to 3.0
    # Measure: settling time, overshoot, steady-state error
    # Should reach target without excessive ringing
```

**Synthetic Data Design**:
- Option A: White noise around mean (models natural variation)
- Option B: Step changes (tests transient response)
- Option C: Real entropy samples from logs (most realistic)

**Files to Test**:
- `models/entropy_scaling.py:59-105` - update() logic
- `models/attn_patch.py:70-78` - parameter initialization

**Complexity**: LOW-MEDIUM - Unit test with statistics, no GPU needed

---

## BUG #3: Inverted Dose-Response

### Test Strategy

**Goal**: Show that temperature changes correlate with entropy increases instead of decreases.

**Challenge**: This is a system-level property, harder to isolate in unit test.

**Approach Options**:

**Option A - Controlled Synthetic Test**:
```python
def test_dose_response_controlled():
    """Test temp effect on entropy in controlled setting."""
    # Create simple attention scenario:
    #   Q, K, V with known structure
    #   e.g., K has one very high similarity, rest low
    # Compute entropy at temp=1.0, temp=0.7, temp=0.9
    # Expected: Lower temp → lower entropy
    # If this passes, bug is in feedback timing, not mechanism
```

**Option B - Empirical Analysis Test**:
```python
def test_dose_response_from_logs():
    """Analyze dose-response from real logs."""
    # Load: logs/qa_1_entropy_attn_entropy_logs_scaled.jsonl
    # For each example: Extract dT[t] and dH[t+1]
    # Bin by |dT| magnitude
    # Plot: mean(dH) vs |dT|
    # Assert: Positive correlation (documents bug)
    # After fix: Should be negative or flat
```

**Option C - Phase Lag Analysis**:
```python
def test_feedback_phase_lag():
    """Measure how long controller response takes to affect entropy."""
    # Inject known entropy perturbation
    # Track: When does temp change? When does entropy respond?
    # Calculate: lag in steps
    # If lag > 1, explains why feedback seems inverted
```

**Files to Analyze**:
- `logs/qa_1_entropy_attn_entropy_logs_scaled.jsonl`
- `logs/entropy_scaling_analysis.ipynb` (existing analysis)
- `models/entropy_scaling.py` - Control law

**Complexity**: MEDIUM-HIGH - Requires either GPU for synthetic tests or log analysis infrastructure

**Note**: This bug might not have a "fix", but rather reveal fundamental limitation of per-token feedback.

---

## BUG #4: Triton vs PyTorch Entropy Inconsistency

### Test Strategy

**Goal**: Verify that Triton (prefill) and PyTorch (decode) compute identical entropy.

**Approach**:
- Create small test case: Q, K, V, temp
- Compute entropy via Triton path (force prefill, N_CTX > 1)
- Compute entropy via PyTorch reference implementation
- Compare: Should match within FP16 tolerance (~1e-3)

**Test Structure**:
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_entropy_triton_vs_pytorch_prefill():
    """Triton kernel should match PyTorch reference in prefill."""
    # Setup: Z=1, H=2, N_CTX=8, HEAD_DIM=64
    # Generate: random Q, K, V
    # Triton: o_t, H_t = entropy_attention(q, k, v, decode=False)
    # PyTorch: H_ref = compute_entropy_pytorch(q, k, v)
    # Assert: allclose(H_t, H_ref, atol=0.1)  # Relaxed for FP16

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_entropy_decode_uses_pytorch():
    """Decode path should use PyTorch (no Triton kernel)."""
    # Setup: N_Q=1, N_KV=128 (decode scenario)
    # Call: entropy_attention(..., decode=True)
    # Verify: Falls back to PyTorch path (check line 552-565)
    # Compute reference independently
    # Assert: Match exactly (both use same code)
```

**Files to Test**:
- `models/entropy_attn_triton.py:240-246` - Triton entropy computation
- `models/entropy_attn_triton.py:552-565` - PyTorch fallback
- `test_kernel.py` - Existing kernel tests (may already cover this?)

**Complexity**: MEDIUM - Requires CUDA, FP16 handling, Triton kernel invocation

**Risk Assessment**: If mismatch found, need to align implementations (likely fix PyTorch to match Triton).

---

## Test Implementation Order

### Phase 1: Easiest Wins (No GPU)
1. ✅ BUG #1 reproduction test
2. ✅ BUG #2 saturation metrics test
3. ✅ BUG #1 fix verification test

### Phase 2: GPU-Dependent Tests
4. BUG #4 entropy consistency test (prefill)
5. BUG #4 entropy consistency test (decode)

### Phase 3: Complex System Tests
6. BUG #2 improved parameters test
7. BUG #3 dose-response analysis (if time permits)

### Phase 4: Integration Tests
8. End-to-end: Reset during evaluation loop
9. End-to-end: Run 5 examples, verify independence
10. Regression: Existing tests still pass

---

## Test File Organization

```
tests/
├── test_controller_unit.py          # BUG #1, #2 unit tests
├── test_entropy_consistency.py      # BUG #4 Triton vs PyTorch
├── test_controller_integration.py   # Multi-example scenarios
├── test_dose_response.py            # BUG #3 analysis (optional)
└── fixtures/
    ├── synthetic_entropy.py         # Generate test data
    └── mock_attention.py            # Mock attention modules
```

Or simpler structure:
```
test_controller.py         # All controller-related tests
test_entropy.py            # Entropy computation tests
test_integration.py        # End-to-end scenarios
```

---

## Success Criteria

### For BUG #1:
- [ ] Test proves state leaks (reproduction)
- [ ] reset() method implemented in EntropyTempController
- [ ] reset_entropy_controller() calls it correctly
- [ ] Test proves state resets properly (verification)

### For BUG #2:
- [ ] Test measures sat_frac > 0.4 with current params (reproduction)
- [ ] Test measures sat_frac < 0.25 with new params (verification)
- [ ] Controller operates mostly within [0.75, 0.98] range

### For BUG #4:
- [ ] Test shows entropy matches within 10% relative error
- [ ] If mismatch found: Document which one is "correct"
- [ ] If mismatch found: Align implementations

---

## Open Questions

1. **Should we test with real model?**
   - Pro: Most realistic, catches integration issues
   - Con: Slow, requires loading weights, harder to debug
   - Decision: Start with mocks, add real model test later

2. **What tolerance for entropy consistency?**
   - FP16 noise: ~1e-3 absolute
   - Algorithm difference: Unknown, need to measure
   - Proposal: Start with 10% relative, tighten if possible

3. **How to generate realistic synthetic entropy?**
   - Option A: Sample from real logs distribution
   - Option B: Parametric model (normal + occasional spikes)
   - Option C: Run actual attention with random inputs

4. **Should we test all layers or just one?**
   - Controller is per-layer, state is independent
   - Testing one layer sufficient for unit tests
   - Integration test should check multi-layer

---

## Next Steps

1. Review this plan with human
2. Start with Phase 1 tests (no GPU needed)
3. Implement BUG #1 tests first (clearest, most critical)
4. Run tests, confirm bugs reproduce
5. Implement fixes
6. Re-run tests, confirm fixes work
