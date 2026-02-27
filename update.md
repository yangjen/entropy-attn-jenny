# Project Direction Update: Streaming Inference-Time Adaptive Decoding (Entropy → Attention-Temperature Control)

## TL;DR
We reposition the project as **streaming / session-level inference-time adaptation**: the controller **does not reset per sample** inside a session and instead carries state across consecutive requests. We **calibrate the starting temperature** on **InfiniBench (validation)** and evaluate on **RULER (test)** using a **sessionized protocol** with warm-up vs mature-phase reporting and multiple random orderings.

---

## Motivation
- We previously observed consistent gains when controller state persisted across samples (no reset), across multiple context lengths.
- With strict per-sample reset (i.i.d. protocol), gains on RULER largely disappeared.
- Synthetic tests/bug analysis indicate: the controller can converge when the target is reachable for a given attention regime, but a single global entropy target can be incompatible across diverse token-level attention regimes. Streaming adaptation may still help by converging to a better operating point for the stream distribution.

---

## Current Approach

### What we control (attention-side temperature)
We intervene by applying **temperature-like scaling inside attention softmax** (Q·K → softmax):

`attn(q, k) = softmax((q k^T) / T)`

- Smaller \(T\) → sharper (more peaked) attention
- Larger \(T\) → flatter attention
- We bound temperature: T in `[T_min, 1.0]`.

### Controller state (what it remembers across items)
The controller is a stateful online control loop attached to attention modules. Within a session, it persists across samples and remembers:
- Current attention temperature `T` (per head)
- EMA of normalized attention entropy (per head)
- Optional prompt-derived target entropy (per head) from the prefill stage
- Health/limits: update step budget, clamping/saturation behavior

---

## Scaling / Update Logic (current implementation)

### Signals
At each decode step, we read the attention entropy for the last token and normalize it by context length:
- `kv_len` = current KV cache length
- `norm = log(kv_len)` (clamped to >= 1)
- `H_norm = H_last / norm`

We maintain EMA smoothing per head:
- `EMA <- beta * EMA + (1 - beta) * H_norm`

### Targeting and error signal (one-sided)
If a target entropy exists (estimated during prefill), we compute a one-sided error:
- `err = max(EMA - target, 0)`

If no target exists:
- `err = EMA` (still nonnegative)

**Implication:** the controller primarily **sharpens** when entropy is above target; it does not actively “relax” when entropy is below target.

### Temperature update (bounded proportional control)
We update temperature with bounded steps:
- `delta = clip(-kp * err, [-max_step, +max_step])`
- `T <- clip(T + delta, [T_min, 1.0])`

---

## Streaming / Session Protocol

### Why sessionization is needed
RULER is normally evaluated i.i.d. (each example independent). Because our method carries state across samples, we must define a streaming protocol:
- **Reset between sessions**, not between samples.
- Within a session, controller state persists across items.

### Warm-up vs mature phase
We expect adaptation to plateau:
- Session length: `K`
- Warm-up length: `W`
- Report:
  - Warm-up performance: items `1..W`
  - **Mature-phase performance**: items `W+1..K` (primary streaming metric)

### Order dependence
Because online adaptation can depend on ordering:
- Run multiple random orderings (e.g., 5–10 shuffles).
- Report mean ± std across orderings (and optionally min/max).

---

## Calibration Plan (InfiniBench → RULER)

### Validation (InfiniBench)
We use **InfiniBench** as validation to tune:
- Initial temperature `T0` (starting operating point), and/or
- A fixed constant temperature `T*` (for baseline S1)

Tuning targets:
- stable behavior (avoid frequent saturation)
- reasonable intervention coverage
- (optionally) validation accuracy as a secondary signal

### Test (RULER)
We treat **RULER as test**, evaluated under the sessionized streaming protocol described above.

---

## Baselines / Ablations (committed)

### Baseline 0: Stateless (standard)
- No controller / no streaming adaptation
- Equivalent to fixed attention temperature `T=1.0`

### S1: Calibrated constant attention temperature (must-have)
- Tune a fixed `T*` on InfiniBench
- Keep it constant during evaluation (disable online updates)
- Purpose: separates “better constant operating point” from “benefit of online adaptation”

### S3: No-EMA ablation (secondary)
- Remove EMA smoothing (use instantaneous `H_norm` each step)
- Keep the same update rule and bounds
- Purpose: tests whether stateful smoothing/memory is necessary for streaming gains

---

## Metrics to Report

### Primary (RULER)
- Accuracy / task score
- Reported by:
  - task (e.g., qa_1, qa_2)
  - context length bucket
  - session position summary: warm-up vs mature phase

### Streaming-specific
- Warm-up curve: accuracy vs position in session
- Robustness: mean ± std across orderings

### Controller health / mechanism
- Intervention coverage (% decode steps where `err > 0` / temp update applied)
- Saturation rate (% time `T` hits bounds)
- Temperature trajectory over session (plateau evidence)

---

## Next Steps
1. Implement sessionization (`K`, `W`) + multiple random orderings in the RULER runner.
2. Add InfiniBench tuning sweep for `T0` and `T*` (log coverage/saturation/stability).
3. Run the baseline/ablation set: Stateless, S1 (fixed tuned), Full streaming controller, S3 (no EMA).
4. Prepare plots: warm-up curve, temperature trajectory, ordering-robust mature-phase performance.
5. (After RULER streaming is stable) extend to an agentic benchmark (e.g., WebArena) where sessions are natural (one trajectory = one session) and the main metric is task success rate.
