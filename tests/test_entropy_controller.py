"""Tests for entropy-based temperature controller.

Tests verify controller behavior under various conditions:
- Control law correctness
- Convergence with delays
- Response to transient spikes
- Robustness to per-token entropy variance
- Isolation of entropy sources (temperature vs semantic patterns)
"""

import pytest
import torch
import numpy as np
from models.entropy_ops import (
    update_ema,
    compute_temperature_delta,
    update_temperature,
    compute_entropy_from_attention_weights,
    compute_controller_health_metrics,
)


def simple_attention_model(temperature, base_entropy=2.5, sensitivity=1.0):
    """Simplified model: lower temperature → lower entropy.

    H = base - sensitivity * (1.0 - T)
    """
    return base_entropy - sensitivity * (1.0 - temperature)


# ============================================================================
# Synthetic Attention Pattern Generators
# ============================================================================

def create_peaked_pattern(kv_len=128, peak_idx=None, sharpness=1.0):
    """Simulate focused attention (e.g., copying a specific token).

    Args:
        kv_len: Number of KV positions
        peak_idx: Where to focus (default: middle)
        sharpness: How sharp the peak (higher = more focused)

    Returns:
        Normalized attention pattern with single peak
    """
    peak_idx = peak_idx if peak_idx is not None else kv_len // 2
    positions = torch.arange(kv_len, dtype=torch.float32)
    distances = torch.abs(positions - peak_idx)
    weights = torch.exp(-sharpness * distances)
    return weights / weights.sum()


def create_uniform_pattern(kv_len=128):
    """Simulate broad context gathering (all positions equally important).

    Returns:
        Uniform attention distribution
    """
    return torch.ones(kv_len) / kv_len


def create_recency_biased_pattern(kv_len=128, decay=0.95):
    """Simulate recency bias (recent tokens more important).

    Args:
        kv_len: Number of KV positions
        decay: Decay factor per position (0.95 = 5% decay per step)

    Returns:
        Exponentially decaying attention pattern
    """
    # Most recent position has highest weight
    weights = torch.tensor([decay ** (kv_len - 1 - i) for i in range(kv_len)])
    return weights / weights.sum()


def create_bimodal_pattern(kv_len=128, mode1_idx=None, mode2_idx=None):
    """Simulate choosing between two options.

    Returns:
        Attention pattern with two peaks of equal weight
    """
    mode1_idx = mode1_idx if mode1_idx is not None else kv_len // 3
    mode2_idx = mode2_idx if mode2_idx is not None else 2 * kv_len // 3

    pattern = torch.zeros(kv_len)
    pattern[mode1_idx] = 0.5
    pattern[mode2_idx] = 0.5
    return pattern


def apply_temperature_to_logits(logits, temperature):
    """Apply temperature scaling to logits before softmax.

    This simulates what happens in the attention kernel:
    - Lower temp → sharper distribution → lower entropy
    - Higher temp → flatter distribution → higher entropy

    Args:
        logits: Pre-softmax scores [kv_len]
        temperature: Scaling factor

    Returns:
        Attention weights after temperature scaling
    """
    return torch.softmax(logits / temperature, dim=-1)


def pattern_to_logits(pattern, temperature=1.0):
    """Invert softmax to get logits that produce given pattern at temperature.

    This allows us to:
    1. Start with desired attention pattern
    2. Extract logits
    3. Apply different temperatures
    4. See how entropy changes

    Args:
        pattern: Target attention distribution
        temperature: Temperature used to generate pattern

    Returns:
        Logits that produce pattern at given temperature
    """
    # Invert softmax: logits = temperature * log(pattern) + constant
    # The constant doesn't matter (softmax is shift-invariant)
    logits = temperature * torch.log(pattern + 1e-9)
    return logits


# ============================================================================
# Tests: Control Law Correctness
# ============================================================================

class TestControlLawCorrectness:
    """Verify the control law itself has correct direction."""

    def test_high_entropy_gives_negative_delta(self):
        """High entropy should produce negative delta (sharpen)."""
        current = torch.tensor([[[3.0]]])
        target = torch.tensor([[[2.0]]])

        delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.01)

        assert delta.item() < 0, "Should decrease temperature when entropy too high"

    def test_larger_error_gives_larger_delta(self):
        """Larger entropy error should produce larger temperature change."""
        target = torch.tensor([[[2.0]]])

        small_err = torch.tensor([[[2.5]]])  # err = 0.5
        large_err = torch.tensor([[[3.0]]])  # err = 1.0

        delta_small = compute_temperature_delta(small_err, target, kp=0.35, max_step=1.0)
        delta_large = compute_temperature_delta(large_err, target, kp=0.35, max_step=1.0)

        assert abs(delta_large.item()) > abs(delta_small.item())


class TestConvergenceWithDelay:
    """Test closed-loop system with measurement delays."""

    def test_converges_with_ideal_response(self):
        """Controller should converge when entropy responds immediately to temp."""
        target_val = 2.2  # Set achievable target
        target = torch.tensor([[[target_val]]])

        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        for step in range(500):  # More steps for convergence
            # Entropy responds IMMEDIATELY to current temperature
            true_entropy = simple_attention_model(temp.item(), base_entropy=2.8, sensitivity=2.0)
            H = torch.tensor([[[true_entropy]]])

            # Update EMA
            ema_entropy = update_ema(ema_entropy, H, beta=0.9) if step > 0 else H

            # Control
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.001)
            temp = update_temperature(temp, delta, 0.7, 1.0)

        # Should converge close to target
        final_entropy = simple_attention_model(temp.item(), base_entropy=2.8, sensitivity=2.0)
        print(f"\nIdeal response convergence:")
        print(f"  Final entropy: {final_entropy:.4f} (target: {target_val:.4f})")
        print(f"  Final temp: {temp.item():.6f}")
        print(f"  Final EMA entropy: {ema_entropy.item():.4f}")
        assert abs(final_entropy - target_val) < 0.1

    def test_converges_with_one_step_delay(self):
        """Controller should still converge with 1-step measurement delay.

        Real system:
        - Step t: Controller updates temp[t] based on measured H[t-1]
        - Step t+1: Attention uses temp[t], produces H[t]
        """
        target_val = 2.2  # Set achievable target
        target = torch.tensor([[[target_val]]])

        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        # Previous entropy (what we measure at step t comes from temp at step t-1)
        prev_entropy = simple_attention_model(1.0, base_entropy=2.8, sensitivity=2.0)

        for step in range(500):  # More steps for convergence with delay
            # Measure entropy from PREVIOUS temperature (1-step delay)
            H_measured = torch.tensor([[[prev_entropy]]])

            # Update EMA based on delayed measurement
            ema_entropy = update_ema(ema_entropy, H_measured, beta=0.9) if step > 0 else H_measured

            # Control based on delayed measurement
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.001)
            temp = update_temperature(temp, delta, 0.7, 1.0)

            # TRUE entropy responds to NEW temperature (but won't be seen until next step)
            prev_entropy = simple_attention_model(temp.item(), base_entropy=2.8, sensitivity=2.0)

        # Even with delay, should converge (may be slower)
        final_entropy = prev_entropy
        print(f"\nDelayed response convergence:")
        print(f"  Final entropy: {final_entropy:.4f} (target: {target_val:.4f})")
        print(f"  Final temp: {temp.item():.6f}")
        print(f"  Final EMA entropy: {ema_entropy.item():.4f}")
        assert abs(final_entropy - target_val) < 0.1


class TestTransientSpikeResponse:
    """Test controller response to transient entropy spikes."""

    def test_transient_spike_causes_delayed_reaction(self):
        """Inject a transient entropy spike and observe controller response.

        Scenario:
        1. System at equilibrium (H ≈ target)
        2. At step 50: external event causes entropy spike
        3. Spike lasts only 1 step (not temperature-related)
        4. Controller reacts by lowering temperature
        5. But spike already gone by time reaction takes effect
        6. Result: temperature too low for subsequent tokens
        """
        target = torch.tensor([[[2.0]]])

        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[2.0]]])  # Start at equilibrium

        temps = []
        measured_H = []

        for step in range(100):
            # Base entropy (what we'd naturally have)
            base_H = simple_attention_model(temp.item(), base_entropy=2.0, sensitivity=1.0)

            # Inject transient spike at step 50
            if step == 50:
                H_measured = 3.5  # Sudden spike
            else:
                H_measured = base_H

            measured_H.append(H_measured)
            H_tensor = torch.tensor([[[H_measured]]])

            # Controller sees the spike and reacts
            ema_entropy = update_ema(ema_entropy, H_tensor, beta=0.9)
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.001)
            temp = update_temperature(temp, delta, 0.7, 1.0)
            temps.append(temp.item())

        # After spike: temperature should decrease (reaction to spike)
        # But spike was transient, so lower temp is now unnecessary
        print(f"\nTransient spike analysis:")
        print(f"  Temp before spike (step 49): {temps[49]:.6f}")
        print(f"  Temp at spike (step 50):     {temps[50]:.6f}")
        print(f"  Temp after spike (step 55):  {temps[55]:.6f}")
        print(f"  Temp recovery (step 70):     {temps[70]:.6f}")

        # Key: temperature drops AFTER spike, even though spike is gone
        assert temps[55] < temps[49], "Temperature drops after transient spike"


class TestNoiseRobustness:
    """Test controller robustness to per-token entropy variance."""

    def test_clean_signal_converges_perfectly(self):
        """With clean signal (no noise), controller should converge deterministically."""
        target_val = 2.3  # In reachable range [2.05, 2.5] - won't saturate
        target = torch.tensor([[[target_val]]])

        np.random.seed(42)
        torch.manual_seed(42)

        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        temps = []
        entropies = []

        for step in range(1000):
            # Entropy = base (from temp) only, no noise
            base_H = simple_attention_model(temp.item(), base_entropy=2.5, sensitivity=1.5)

            entropies.append(base_H)
            H_tensor = torch.tensor([[[base_H]]])

            # Controller with production parameters
            ema_entropy = update_ema(ema_entropy, H_tensor, beta=0.9) if step > 0 else H_tensor
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.0005)
            temp = update_temperature(temp, delta, 0.7, 1.0)
            temps.append(temp.item())

        # Check convergence
        final_entropy = entropies[-1]
        convergence_error = abs(final_entropy - target_val)

        print(f"\nClean signal (no noise) - Deterministic convergence:")
        print(f"  Final entropy: {final_entropy:.4f}")
        print(f"  Target: {target_val:.4f}")
        print(f"  Convergence error: {convergence_error:.6f}")
        print(f"  Expected: Perfect convergence (error ≈ 0)")

        # With clean signal, should converge to target within tight tolerance
        assert convergence_error < 0.01, f"Failed to converge: error = {convergence_error:.6f}"

    def test_high_noise_degrades_control_effectiveness(self):
        """With noisy signal, controller effectiveness degrades.

        Demonstrates that per-token entropy variance reduces control effectiveness.
        """
        target = torch.tensor([[[2.0]]])
        noise_std = 0.8  # High noise to demonstrate effect

        np.random.seed(42)
        torch.manual_seed(42)

        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        temps = []
        entropies = []

        for step in range(400):
            # Entropy = base (from temp) + noise (from token)
            base_H = simple_attention_model(temp.item(), base_entropy=2.5, sensitivity=1.5)
            noisy_H = base_H + np.random.normal(0, noise_std)

            entropies.append(noisy_H)
            H_tensor = torch.tensor([[[noisy_H]]])

            # Controller with production parameters
            ema_entropy = update_ema(ema_entropy, H_tensor, beta=0.9) if step > 0 else H_tensor
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.0005)
            temp = update_temperature(temp, delta, 0.7, 1.0)
            temps.append(temp.item())

        # Compute dose-response
        dT = np.diff(temps)
        dH = np.diff(entropies)

        # Correlation
        overall_corr = np.corrcoef(dT, dH)[0, 1]

        print(f"\nNoisy signal (noise_std={noise_std}):")
        print(f"  Correlation(ΔT, ΔH): {overall_corr:+.4f}")
        print(f"  Expected: near zero or positive (degraded due to noise)")
        print(f"  This shows controller chasing per-token entropy variance")

        # With noise, correlation may be positive (inverted) or near zero
        # This shows controller is ineffective or counterproductive
        print(f"\n  Result: {'INVERTED (positive)' if overall_corr > 0 else 'Weakened (near zero)'}")

    @pytest.mark.xfail(reason="Known issue: Controller degrades with high per-token entropy variance")
    def test_controller_robust_to_token_entropy_variance(self):
        """Controller should converge despite per-token entropy variance.

        This test CURRENTLY FAILS. It should PASS after implementing noise-robust control.

        Tests that controller maintains convergence quality across different
        noise levels, demonstrating robustness to per-token entropy variance.
        """
        target_val = 2.3  # Same reachable target as clean signal test
        target = torch.tensor([[[target_val]]])

        noise_levels = [0.0, 0.4, 0.8]  # Low, medium, high noise
        n_trials = 20
        n_steps = 300
        convergence_threshold = 0.3  # Within 0.3 of target = success

        for noise_std in noise_levels:
            convergence_errors = []

            for trial in range(n_trials):
                seed = 42 + trial
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Random initial conditions
                temp_init = 0.85 + 0.15 * np.random.rand()
                temp = torch.tensor([[[temp_init]]])
                ema_entropy = torch.tensor([[[0.0]]])

                entropies = []

                for step in range(n_steps):
                    # Base entropy + per-token variance
                    base_H = simple_attention_model(
                        temp.item(),
                        base_entropy=2.5,  # Same as clean test
                        sensitivity=1.5
                    )
                    noisy_H = base_H + np.random.normal(0, noise_std)
                    entropies.append(noisy_H)

                    H_tensor = torch.tensor([[[noisy_H]]])
                    ema_entropy = update_ema(ema_entropy, H_tensor, beta=0.9) if step > 0 else H_tensor
                    delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.0005)
                    temp = update_temperature(temp, delta, 0.7, 1.0)

                # Measure final convergence (last 50 steps)
                final_error = np.mean(np.abs(np.array(entropies[-50:]) - target_val))
                convergence_errors.append(final_error)

            # Statistics
            mean_error = np.mean(convergence_errors)
            success_rate = np.mean([e < convergence_threshold for e in convergence_errors])

            print(f"\nNoise σ={noise_std}: Error={mean_error:.4f}, Success={success_rate:.1%}")

            # Controller should maintain >80% success rate even with high noise
            assert success_rate > 0.80, \
                f"Controller failed with noise σ={noise_std}: only {success_rate:.1%} converged " \
                f"(mean error: {mean_error:.4f}). Expected >80% success rate."


# ============================================================================
# Tests: Entropy Source Decomposition
# ============================================================================

# Pattern generators for parameterized tests (named for clear pytest output)
def gen_peaked_sharp():
    return create_peaked_pattern(128, sharpness=5.0)

def gen_peaked_moderate():
    return create_peaked_pattern(128, sharpness=2.0)

def gen_recency():
    return create_recency_biased_pattern(128, decay=0.95)

def gen_bimodal():
    return create_bimodal_pattern(128)

def gen_uniform():
    return create_uniform_pattern(128)


class TestPureTemperatureControl:
    """Test controller with constant Q·K pattern (semantic variance = 0).

    Key insight: Each pattern has a reachable entropy range determined by Q·K.
    Temperature can only modulate within ~0.35 nats of that range.

    These tests show: controller WORKS when pattern matches target,
    but FAILS when pattern doesn't support the target entropy.
    """

    # Control case: target that works for peaked_moderate pattern
    CONTROL_TARGET = 0.634  # Middle of peaked_moderate range [0.44, 0.82]

    def test_control_case_peaked_moderate_converges(self):
        """Control case: peaked_moderate pattern with compatible target.

        This is the IDEAL scenario - pattern and target are compatible.
        Controller should converge perfectly.

        All other tests use this SAME target but different patterns.
        """
        base_pattern = gen_peaked_moderate()
        base_logits = pattern_to_logits(base_pattern)

        # Verify this target is within reachable range
        H_min = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, 0.7)
        ).item()
        H_max = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, 1.0)
        ).item()

        assert H_min <= self.CONTROL_TARGET <= H_max, \
            f"Control target {self.CONTROL_TARGET} not in range [{H_min}, {H_max}]"

        # Run controller
        target = torch.tensor([[[self.CONTROL_TARGET]]])
        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        for step in range(500):
            attn = apply_temperature_to_logits(base_logits, temp.item())
            H = compute_entropy_from_attention_weights(attn)

            H_tensor = torch.tensor([[[H.item()]]])
            ema_entropy = update_ema(ema_entropy, H_tensor, beta=0.9) if step > 0 else H_tensor
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.0005)
            temp = update_temperature(temp, delta, 0.7, 1.0)

        final_entropy = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, temp.item())
        ).item()
        convergence_error = abs(final_entropy - self.CONTROL_TARGET)

        print(f"\nCONTROL CASE - peaked_moderate:")
        print(f"  Reachable range: [{H_min:.4f}, {H_max:.4f}]")
        print(f"  Target: {self.CONTROL_TARGET:.4f}")
        print(f"  Final entropy: {final_entropy:.4f}")
        print(f"  Convergence error: {convergence_error:.6f}")
        print(f"  Result: CONVERGED ✓")

        assert convergence_error < 0.02, f"Control case should converge: error = {convergence_error}"

    @pytest.mark.xfail(reason="Target from peaked_moderate is unreachable for other patterns", strict=True)
    @pytest.mark.parametrize("pattern_name,pattern_gen", [
        ("peaked_sharp", gen_peaked_sharp),
        ("recency", gen_recency),
        ("bimodal", gen_bimodal),
        ("uniform", gen_uniform),
    ])
    def test_other_patterns_fail_to_reach_control_target(self, pattern_name, pattern_gen):
        """Other patterns cannot reach peaked_moderate's target.

        Pattern ranges:
        - peaked_sharp: [0.01, 0.08] - target too high
        - recency: [3.61, 3.96] - target too low
        - bimodal: [0.69, 0.69] - target slightly too low
        - uniform: [4.85, 4.85] - target way too low

        Target: 0.634 (from peaked_moderate)
        Expected: These tests FAIL (marked with xfail)
        """
        base_pattern = pattern_gen()
        base_logits = pattern_to_logits(base_pattern)

        H_min = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, 0.7)
        ).item()
        H_max = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, 1.0)
        ).item()

        # Run controller with control target
        target = torch.tensor([[[self.CONTROL_TARGET]]])
        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        for step in range(500):
            attn = apply_temperature_to_logits(base_logits, temp.item())
            H = compute_entropy_from_attention_weights(attn)

            H_tensor = torch.tensor([[[H.item()]]])
            ema_entropy = update_ema(ema_entropy, H_tensor, beta=0.9) if step > 0 else H_tensor
            delta = compute_temperature_delta(ema_entropy, target, kp=0.35, max_step=0.0005)
            temp = update_temperature(temp, delta, 0.7, 1.0)

        final_entropy = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, temp.item())
        ).item()
        convergence_error = abs(final_entropy - self.CONTROL_TARGET)

        print(f"\n{pattern_name}:")
        print(f"  Reachable range: [{H_min:.4f}, {H_max:.4f}]")
        print(f"  Target: {self.CONTROL_TARGET:.4f} (from peaked_moderate)")
        print(f"  Final entropy: {final_entropy:.4f}")
        print(f"  Convergence error: {convergence_error:.4f}")
        print(f"  Result: FAILED ✗ (target unreachable)")

        # Assertion expects convergence but will FAIL (xfail catches this)
        assert convergence_error < 0.02, \
            f"{pattern_name} should converge to {self.CONTROL_TARGET} but cannot (expected xfail)"




class TestSemanticVariance:
    """Measure natural entropy variance from Q·K patterns alone.

    Key finding: Semantic variance (4.4 nats) >> Temperature effect (0.35 nats).
    This is why single-target control from prompt fails in decode.
    """

    def test_entropy_range_across_pattern_types(self):
        """Different semantic patterns have very different base entropies.

        This test is deterministic - no randomness.
        """
        kv_len = 128

        patterns = {
            "peaked": create_peaked_pattern(kv_len, sharpness=5.0),
            "moderate_peak": create_peaked_pattern(kv_len, sharpness=1.0),
            "recency": create_recency_biased_pattern(kv_len, decay=0.95),
            "bimodal": create_bimodal_pattern(kv_len),
            "uniform": create_uniform_pattern(kv_len),
        }

        entropies = {}
        for name, pattern in patterns.items():
            H = compute_entropy_from_attention_weights(pattern)
            entropies[name] = H.item()

        semantic_range = max(entropies.values()) - min(entropies.values())

        print(f"\nEntropy by pattern type (fixed T=1.0):")
        for name in ["peaked", "moderate_peak", "recency", "bimodal", "uniform"]:
            print(f"  {name:15s}: H = {entropies[name]:.4f}")
        print(f"\nSemantic variance range: {semantic_range:.4f} nats")

        # Semantic patterns cause large entropy variation
        assert semantic_range > 4.0, f"Pattern type should dominate entropy: range = {semantic_range}"

    def test_temperature_effect_vs_semantic_variance(self):
        """Quantify the ratio: semantic variance / temperature effect.

        This is the KEY metric showing why controller struggles.
        Deterministic test - always produces same ratio.
        """
        kv_len = 128

        # Measure temperature effect on one pattern
        base_pattern = create_recency_biased_pattern(kv_len, decay=0.95)
        base_logits = pattern_to_logits(base_pattern)

        H_at_min_temp = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, 0.7)
        ).item()
        H_at_max_temp = compute_entropy_from_attention_weights(
            apply_temperature_to_logits(base_logits, 1.0)
        ).item()

        temperature_effect = H_at_max_temp - H_at_min_temp

        # Measure semantic variance across patterns (at fixed temp)
        patterns = [
            create_peaked_pattern(kv_len, sharpness=3.0),
            create_recency_biased_pattern(kv_len, decay=0.95),
            create_uniform_pattern(kv_len),
        ]

        entropies = [compute_entropy_from_attention_weights(p).item() for p in patterns]
        semantic_variance = max(entropies) - min(entropies)

        ratio = semantic_variance / temperature_effect

        print(f"\nEffect size comparison:")
        print(f"  Temperature effect (T: 0.7→1.0): ΔH = {temperature_effect:.4f}")
        print(f"  Semantic variance (pattern types): ΔH = {semantic_variance:.4f}")
        print(f"  Ratio (semantic/temp):              {ratio:.2f}x")
        print(f"\nConclusion: Semantic changes are {ratio:.1f}x larger than temperature control.")

        # Semantic variance should be much larger (>10x)
        assert ratio > 10.0, f"Semantic variance should dominate: ratio = {ratio:.2f}x"

    def test_reachable_ranges_by_pattern_type(self):
        """Show each pattern has a narrow reachable range.

        Demonstrates that temperature can only modulate ~0.3-0.4 nats.
        This table is critical for understanding unreachable targets.
        """
        kv_len = 128

        patterns = {
            "peaked (sharp=5)": create_peaked_pattern(kv_len, sharpness=5.0),
            "peaked (sharp=2)": create_peaked_pattern(kv_len, sharpness=2.0),
            "recency (0.95)": create_recency_biased_pattern(kv_len, decay=0.95),
            "uniform": create_uniform_pattern(kv_len),
        }

        print(f"\nReachable entropy ranges by pattern (T: 0.7 → 1.0):")
        print(f"{'Pattern':20s}  {'H_min':>7s}  {'H_max':>7s}  {'Range':>7s}")
        print("=" * 50)

        for name, pattern in patterns.items():
            logits = pattern_to_logits(pattern)
            H_min = compute_entropy_from_attention_weights(
                apply_temperature_to_logits(logits, 0.7)
            ).item()
            H_max = compute_entropy_from_attention_weights(
                apply_temperature_to_logits(logits, 1.0)
            ).item()
            temp_range = H_max - H_min

            print(f"{name:20s}  {H_min:7.4f}  {H_max:7.4f}  {temp_range:7.4f}")

        print(f"\nKey insight: Each pattern has narrow reachable range (~0.3-0.4 nats).")
        print(f"Single target from prompt is unreachable for most decode patterns.")

