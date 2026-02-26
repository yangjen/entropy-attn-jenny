# Project Direction Update: Streaming Inference-Time Adaptive Decoding (Entropy → Attention-Temperature Control)

## Summary
We reposition the project as **streaming / session-level inference-time adaptation**: the controller maintains state across consecutive requests (no per-sample reset) and converges to a stable operating regime (“plateau”). This better matches the intended deployment target (multi-turn / agentic workflows) than i.i.d. per-item evaluation.

---

## Current Approach

### What is controlled
We intervene by applying **temperature-like scaling to attention logits (Q·K scores)** during decoding.

Conceptually:
- Attention weights are computed as:
  \[
  \text{attn}(q,k) = \text{softmax}\left(\frac{qk^\top}{T}\right)
  \]
- When the risk signal (e.g., attention entropy) indicates uncertainty, we reduce `T` (sharpen attention). Otherwise, we relax toward `T ≈ 1.0`.
- `T` is bounded to avoid extreme behavior:
  - `T ∈ [T_min, 1.0]` (and we track saturation at bounds).

### Controller state (what it remembers)
The controller is a stateful online control loop that persists across items within a session. It remembers:
- Current temperature `T`
- Running statistics of the risk signal (e.g., **EMA** of entropy / uncertainty)
- Step/update budget info (e.g., max update size / max adjustments)
- Optional health stats: saturation counts, intervention counts

Because we **do not reset per sample**, these states carry from item `i` → `i+1`, enabling session-level adaptation.

---

## Why this direction
- We previously observed consistent gains when controller state persisted across items.
- With strict per-sample reset (i.i.d. protocol), gains on RULER largely disappeared.
- Bug analysis suggests a key limitation: **temperature has limited control authority within a fixed attention pattern**, while real decoding exhibits **multiple attention regimes** (peaked/copy, recency, uniform-like, etc.). A single global target can be unreachable for many regimes, leading to saturation.  
Streaming adaptation may still help by converging to a better operating point for the workload distribution, even if per-token setpoint control is imperfect.

---

## Calibration (model-specific starting point)
To avoid test leakage while keeping streaming adaptation:
- Tune a model-specific initial operating point on a **separate long-context validation dataset** (e.g., InfiniBench):
  - initial temperature `T0` **or**
  - entropy anchor `H*` (if the controller uses an entropy reference)
- Then evaluate on **RULER as test**, running in streaming mode (no per-sample reset).

---

## Evaluation Plan (RULER as a streaming proxy)

### Sessionized protocol
RULER is normally i.i.d., so we define a streaming protocol:
- Split the evaluation set into **sessions** of length `K`.
- Reset controller **between sessions only**.
- Within a session, controller state persists across items.

### Warm-up vs mature phase
We expect adaptation to plateau:
- Define warm-up length `W`.
- Report:
  - **Warm-up performance**: items `1..W`
  - **Mature performance**: items `W+1..K` (primary streaming number)

### Order dependence
Because streaming systems can be order-sensitive:
- Run multiple random orderings (e.g., 5–10 shuffles).
- Report mean ± std for mature-phase accuracy.

### Metrics to report
- Primary: **accuracy/task score** (per task, per context length bucket)
- Streaming-specific:
  - warm-up curve (accuracy vs position in session)
  - temperature trajectory (plateau evidence)
- Controller health:
  - intervention coverage
  - saturation rate (`T` at `T_min` or `1.0`)
  - stability/oscillation indicators

### Baselines
- Stateless baseline: fixed decoding / fixed `T=1.0`
- (Recommended) Simple streaming baseline: naive EMA-based global temperature update  
  to show gains are not merely from “having state.”

---

## Next Steps
1. Implement sessionization (`K`, `W`) + multiple orderings; log warm-up curves and mature-phase metrics.
2. Run validation tuning for `T0` or `H*` (track coverage/saturation to avoid degenerate settings).
3. Ablations:
   - tuned start + streaming updates (full)
   - tuned start + no updates (fixed)
   - untuned start + streaming updates
4. Extend to agentic benchmark (e.g., WebArena):
   - session = one trajectory
   - metrics: task success rate, invalid action rate, retries/backtracking, groundedness vs visited evidence.
