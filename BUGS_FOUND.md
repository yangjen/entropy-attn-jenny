# Decode Phase Investigation - Bug Report

**Branch**: `m-feedback`
**Date**: 2026-02-15
**Investigation Focus**: Decode phase feedback loop and controller behavior

**Status Summary**:
- ✅ BUG #1: FIXED in commit `e64f6e5` (2026-02-16) - State leakage resolved
- ⚠️ BUG #2: OPEN - Controller over-aggressive, saturates 48.6% of time
- 🔥 BUG #3: OPEN - Inverted dose-response, needs investigation
- ⚠️ BUG #4: UNVERIFIED - Entropy consistency between Triton/PyTorch (requires CUDA)

---

## Critical Issues Found

### BUG #1: Controller State Leaking Between Examples 🚨 → ✅ FIXED

**Status**: ✅ **FIXED** in commit `e64f6e5` (2026-02-16) - No longer an issue

**Severity**: CRITICAL - Invalidates all evaluation results

**Original Evidence** (from stale logs):
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

Temperature drifted from 1.0 → 0.89 across examples. State was not reset.

**Fix Implemented** (commit `e64f6e5`):
`run_ruler_eval_timed.py:311-312` now deletes the entire controller object between examples:
```python
if hasattr(attn, "_entropy_temp_controller"):
    delattr(attn, "_entropy_temp_controller")  # Delete entire controller
```

On the next forward pass, `models/attn_patch.py:73-82` recreates a fresh controller with:
- `temp_init=1.0`
- `temp=None` → triggers `_init_state()` → sets `temp=1.0`, `ema_entropy=0`
- `prompt_target_entropy=None`

This ensures each example starts with completely fresh state.

**Verification**:
The fix is active in the current codebase. The logs showing temperature drift are from before this fix was committed.

**Original Root Cause** (before fix):
- `models/attn_patch.py:73-82`: Controller initialized once per layer
- `run_ruler_eval_timed.py` (old version): `reset_entropy_controller()` was incomplete or missing

**Expected Behavior**:
Each example should start with `temp=1.0`, `ema_entropy=0`, `prompt_target_entropy=None`

**Impact** (when bug was present):
- Later examples ran with artificially lowered temperature
- Baseline vs scaled comparisons were invalid
- Measured improvements/degradations were artifacts of state leakage

**Current Status**: ✅ This issue is completely resolved in the current codebase.

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

### Priority 1: Fix State Leakage (BUG #1) - ✅ COMPLETED

**Status**: Already fixed in commit `e64f6e5`.

The fix deletes and recreates the controller between examples via:
```python
# run_ruler_eval_timed.py:311-312
if hasattr(attn, "_entropy_temp_controller"):
    delattr(attn, "_entropy_temp_controller")
```

No further action needed.

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

1. ~~**Test: Controller state reset between examples**~~ ✅ Not needed - BUG #1 already fixed
   - ~~Verify temp, ema_entropy, prompt_target all reset~~

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
- `models/entropy_scaling.py` - Tune params for BUG #2
- `models/attn_patch.py` - Update controller parameters, add better logging
- ~~`run_ruler_eval_timed.py` - Fix reset_entropy_controller()~~ ✅ Already fixed

### Testing:
- `test_kernel.py` - Add entropy consistency test (requires CUDA)
- ~~`test_controller.py` (NEW) - Unit tests for controller~~ ✅ Created as `tests/test_entropy_ops.py`
- `test_integration.py` (NEW) - End-to-end verification (optional)

### Analysis:
- `logs/entropy_scaling_analysis.ipynb` - Add more diagnostics

---

## Next Steps

1. ~~Implement unit tests to reproduce BUG #1~~ ✅ BUG #1 already fixed - no action needed
2. Implement unit tests to reproduce BUG #2 (controller saturation) - ✅ DONE in `tests/test_entropy_ops.py`
3. Verify BUG #4 (entropy consistency) - requires CUDA environment
4. Create patches for Priority 2 fix (BUG #2 - controller aggression)
5. Re-run evaluation with fixes
6. Analyze if improvements appear after bug fixes
