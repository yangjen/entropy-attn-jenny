"""Unit tests for entropy_ops.py pure functions.

Tests cover core research logic with synthetic data, enabling fast iteration
and debugging without requiring full model or GPU.

Run with: pytest test_entropy_ops.py -v
"""

import pytest
import torch
import numpy as np
from models.entropy_ops import (
    normalize_entropy,
    compute_entropy_from_attention_weights,
    compute_target_from_tail,
    update_ema,
    compute_temperature_delta,
    update_temperature,
    compute_controller_health_metrics,
)


class TestNormalizeEntropy:
    """Test entropy normalization across different sequence lengths."""

    def test_normalization_range(self):
        """Normalized entropy should be in reasonable range for typical inputs."""
        entropy = torch.tensor([2.0, 3.0, 4.0])
        kv_len = 100

        normalized = normalize_entropy(entropy, kv_len)

        # For typical attention entropy (H ~ 2-4), normalized should be in [0, 1]
        assert torch.all(normalized >= 0)
        assert torch.all(normalized <= 1.5)  # Allow slight overshoot for peaked distributions

    def test_edge_case_kv_len_1(self):
        """Should handle kv_len=1 without division by zero."""
        entropy = torch.tensor([1.0])

        # Should not raise error
        normalized = normalize_entropy(entropy, kv_len=1)

        # With kv_len=1, norm is clamped to 1.0, so entropy unchanged
        assert torch.allclose(normalized, entropy)

    def test_longer_sequences_decrease_normalized_entropy(self):
        """Same raw entropy should normalize to smaller value for longer sequences."""
        entropy = torch.tensor([3.0])

        norm_short = normalize_entropy(entropy, kv_len=10)
        norm_long = normalize_entropy(entropy, kv_len=1000)

        assert norm_short > norm_long


class TestComputeEntropyFromAttentionWeights:
    """Test Shannon entropy computation."""

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have maximum entropy = log(N)."""
        N = 10
        uniform = torch.ones(N) / N

        entropy = compute_entropy_from_attention_weights(uniform)

        # Should be close to log(N)
        expected = torch.log(torch.tensor(float(N)))
        assert torch.allclose(entropy, expected, atol=1e-6)

    def test_peaked_distribution_low_entropy(self):
        """One-hot distribution should have minimal entropy."""
        peaked = torch.zeros(10)
        peaked[0] = 1.0

        entropy = compute_entropy_from_attention_weights(peaked)

        # Should be close to 0
        assert entropy < 0.01

    def test_batch_processing(self):
        """Should handle batched inputs correctly."""
        # [batch=2, seq_len=5]
        attn = torch.softmax(torch.randn(2, 5), dim=-1)

        entropy = compute_entropy_from_attention_weights(attn)

        assert entropy.shape == (2,)
        assert torch.all(entropy > 0)  # Non-zero for random distributions


class TestComputeTargetFromTail:
    """Test target entropy extraction from prompt tail."""

    def test_basic_mean_with_no_trimming(self):
        """With trim_ratio=0, should return simple mean of tail."""
        # [Z=1, H=2, seq_len=100]
        entropy = torch.ones(1, 2, 100) * 2.5

        target = compute_target_from_tail(entropy, tail_length=50, trim_ratio=0.0)

        assert target.shape == (1, 2, 1)
        assert torch.allclose(target, torch.tensor([[[2.5]], [[2.5]]]))

    def test_trimmed_mean_removes_outliers(self):
        """Trimmed mean should be robust to outliers."""
        # Create sequence with outliers
        entropy = torch.ones(1, 1, 100) * 2.0
        entropy[0, 0, -5:] = 10.0  # Add high outliers at end
        entropy[0, 0, -10:-5] = 0.1  # Add low outliers

        target_no_trim = compute_target_from_tail(entropy, tail_length=50, trim_ratio=0.0)
        target_trim = compute_target_from_tail(entropy, tail_length=50, trim_ratio=0.2)

        # Trimmed mean should be closer to 2.0 than untrimmed
        assert torch.abs(target_trim - 2.0) < torch.abs(target_no_trim - 2.0)

    def test_short_sequence_uses_full_length(self):
        """If sequence shorter than tail_length, should use full sequence."""
        entropy = torch.ones(1, 1, 30) * 3.0

        target = compute_target_from_tail(entropy, tail_length=256, trim_ratio=0.0)

        # Should use all 30 tokens
        assert torch.allclose(target, torch.tensor([[[3.0]]]))

    def test_trim_ratio_clamping(self):
        """trim_ratio should be clamped to [0, 0.49]."""
        entropy = torch.randn(1, 1, 100)

        # Should not raise error with invalid trim_ratio
        target_neg = compute_target_from_tail(entropy, trim_ratio=-0.1)
        target_high = compute_target_from_tail(entropy, trim_ratio=0.6)

        # Both should succeed (clamped internally)
        assert target_neg.shape == (1, 1, 1)
        assert target_high.shape == (1, 1, 1)


class TestUpdateEMA:
    """Test exponential moving average."""

    def test_beta_zero_no_memory(self):
        """With beta=0, EMA should equal new value (no memory)."""
        ema_prev = torch.tensor([1.0])
        value_new = torch.tensor([5.0])

        ema_updated = update_ema(ema_prev, value_new, beta=0.0)

        assert torch.allclose(ema_updated, value_new)

    def test_beta_one_no_update(self):
        """With beta=1, EMA should not change (full memory)."""
        ema_prev = torch.tensor([1.0])
        value_new = torch.tensor([5.0])

        ema_updated = update_ema(ema_prev, value_new, beta=1.0)

        assert torch.allclose(ema_updated, ema_prev)

    def test_typical_beta_smoothing(self):
        """With beta=0.9, should provide strong smoothing."""
        ema_prev = torch.tensor([1.0])
        value_new = torch.tensor([2.0])

        ema_updated = update_ema(ema_prev, value_new, beta=0.9)

        # Should be 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        assert torch.allclose(ema_updated, torch.tensor([1.1]))

    def test_convergence_to_constant(self):
        """EMA should converge to constant signal."""
        ema = torch.tensor([0.0])
        target = 5.0
        beta = 0.9

        for _ in range(100):
            ema = update_ema(ema, torch.tensor([target]), beta=beta)

        # Should be close to target after 100 steps
        assert torch.allclose(ema, torch.tensor([target]), atol=0.1)

    def test_valid_mask_selective_update(self):
        """Only positions with valid_mask=True should update."""
        ema_prev = torch.tensor([1.0, 2.0, 3.0])
        value_new = torch.tensor([10.0, 20.0, 30.0])
        mask = torch.tensor([True, False, True])

        ema_updated = update_ema(ema_prev, value_new, beta=0.5, valid_mask=mask)

        # Only positions 0 and 2 should update
        assert torch.allclose(ema_updated[0], torch.tensor(5.5))  # (1+10)/2
        assert torch.allclose(ema_updated[1], torch.tensor(2.0))  # Unchanged
        assert torch.allclose(ema_updated[2], torch.tensor(16.5))  # (3+30)/2


class TestComputeTemperatureDelta:
    """Test core control law - MOST CRITICAL for understanding bugs."""

    def test_high_entropy_sharpens(self):
        """When H_current > H_target, should decrease temperature (sharpen)."""
        current = torch.tensor([[[3.0]]])
        target = torch.tensor([[[2.0]]])

        delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05)

        # Error = 3.0 - 2.0 = 1.0 > 0
        # Delta = -0.35 * 1.0 = -0.35, clamped to -0.05
        assert delta < 0, "Should sharpen (negative delta)"
        assert torch.allclose(delta, torch.tensor([[[-0.05]]]))  # Saturated at max_step

    def test_low_entropy_relaxes_when_allowed(self):
        """When H_current < H_target and allow_increase=True, should increase temp."""
        current = torch.tensor([[[1.0]]])
        target = torch.tensor([[[2.0]]])

        delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05, allow_increase=True)

        # Error = 1.0 - 2.0 = -1.0 < 0
        # Delta = -0.35 * (-1.0) = 0.35, clamped to 0.05
        assert delta > 0, "Should relax (positive delta) when allowed"
        assert torch.allclose(delta, torch.tensor([[[0.05]]]))

    def test_low_entropy_no_action_by_default(self):
        """When H_current < H_target and allow_increase=False (default), should do nothing."""
        current = torch.tensor([[[1.0]]])
        target = torch.tensor([[[2.0]]])

        delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05, allow_increase=False)

        # Error = -1.0, clamped to 0.0 by allow_increase=False
        # Delta = -0.35 * 0.0 = 0
        assert torch.allclose(delta, torch.tensor([[[0.0]]])), "Should not act when entropy too low"

    def test_proportional_response(self):
        """Small errors should produce proportional (not saturated) response."""
        current = torch.tensor([[[2.01]]])
        target = torch.tensor([[[2.00]]])

        delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05)

        # Error = 0.01
        # Delta = -0.35 * 0.01 = -0.0035 (not saturated)
        expected_delta = -0.35 * 0.01
        assert torch.allclose(delta, torch.tensor([[[expected_delta]]]))
        assert torch.abs(delta) < 0.05, "Should not saturate for small error"

    def test_dose_response_curve(self):
        """Test relationship between error magnitude and delta magnitude.

        This is KEY for understanding BUG #3 (inverted dose-response).
        In isolation, larger errors should produce larger deltas (up to saturation).
        """
        target = torch.tensor([[[2.0]]])
        errors = [0.01, 0.05, 0.1, 0.5, 1.0]
        kp = 0.35
        max_step = 0.05

        deltas = []
        for err in errors:
            current = torch.tensor([[[2.0 + err]]])
            delta = compute_temperature_delta(current, target, kp=kp, max_step=max_step)
            deltas.append(abs(delta.item()))

        # Should be monotonically increasing until saturation
        assert deltas[0] < deltas[1] < deltas[2], "Should increase with error"
        assert abs(deltas[3] - max_step) < 1e-6, "Should saturate at max_step for large errors"
        assert abs(deltas[4] - max_step) < 1e-6, "Should stay saturated"

    def test_kp_gain_scaling(self):
        """Higher kp should produce larger response."""
        current = torch.tensor([[[2.5]]])
        target = torch.tensor([[[2.0]]])

        delta_low_gain = compute_temperature_delta(current, target, kp=0.1, max_step=0.1)
        delta_high_gain = compute_temperature_delta(current, target, kp=0.5, max_step=0.1)

        assert torch.abs(delta_high_gain) > torch.abs(delta_low_gain)


class TestUpdateTemperature:
    """Test temperature update with bounds."""

    def test_normal_update(self):
        """Should add delta when within bounds."""
        temp = torch.tensor([[[0.9]]])
        delta = torch.tensor([[[-0.05]]])

        temp_new = update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)

        assert torch.allclose(temp_new, torch.tensor([[[0.85]]]))

    def test_clamp_at_lower_bound(self):
        """Should clamp at temp_min."""
        temp = torch.tensor([[[0.72]]])
        delta = torch.tensor([[[-0.05]]])

        temp_new = update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)

        assert torch.allclose(temp_new, torch.tensor([[[0.7]]]))

    def test_clamp_at_upper_bound(self):
        """Should clamp at temp_max."""
        temp = torch.tensor([[[0.98]]])
        delta = torch.tensor([[[0.05]]])

        temp_new = update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)

        assert torch.allclose(temp_new, torch.tensor([[[1.0]]]))


class TestControllerHealthMetrics:
    """Test diagnostic metrics computation."""

    def test_no_saturation_smooth_trajectory(self):
        """Smooth trajectory should have low saturation and oscillation."""
        temps = torch.linspace(0.75, 0.95, 100)

        metrics = compute_controller_health_metrics(temps, temp_min=0.7, temp_max=1.0)

        assert metrics["temp_sat_frac"] < 0.01, "Should not saturate"
        assert metrics["oscillation"] < 0.05, "Should not oscillate"
        assert metrics["temp_range"] > 0.15, "Should use significant range"

    def test_high_saturation_at_bounds(self):
        """Trajectory at bounds should have high saturation."""
        # Alternate between bounds
        temps = torch.tensor([0.7, 1.0] * 50)

        metrics = compute_controller_health_metrics(temps, temp_min=0.7, temp_max=1.0)

        assert metrics["temp_sat_frac"] > 0.9, "Should be at bounds most of time"
        assert metrics["oscillation"] > 0.9, "Should oscillate every step"

    def test_realistic_buggy_controller(self):
        """Simulate controller with BUG #2 characteristics."""
        # Generate trajectory that saturates frequently
        np.random.seed(42)
        temps_list = [1.0]  # Start at max

        for _ in range(100):
            # Random walk with bias toward bounds
            change = np.random.choice([-0.05, 0.05])  # max_step sized changes
            new_temp = np.clip(temps_list[-1] + change, 0.7, 1.0)
            temps_list.append(new_temp)

        temps = torch.tensor(temps_list)

        metrics = compute_controller_health_metrics(temps, temp_min=0.7, temp_max=1.0)

        print(f"\nBuggy Controller Metrics:")
        print(f"  temp_sat_frac: {metrics['temp_sat_frac']:.3f}")
        print(f"  oscillation: {metrics['oscillation']:.3f}")
        print(f"  temp_range: {metrics['temp_range']:.3f}")

        # Should exhibit BUG #2 characteristics
        assert metrics["temp_sat_frac"] > 0.3, "Should show high saturation"


class TestIntegratedControlLoop:
    """Test complete control loop with synthetic entropy sequences."""

    def test_controller_tracks_constant_target(self):
        """Controller should stabilize near constant target."""
        # Synthetic entropy sequence with noise around target
        target = torch.tensor([[[2.0]]])
        kp = 0.15  # Reduced gain
        ema_beta = 0.9
        max_step = 0.0005

        # Start with temp=1.0
        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[0.0]]])

        temps = [temp.item()]

        # Generate noisy entropy around target
        np.random.seed(42)
        for step in range(100):
            entropy_noisy = target + 0.2 * torch.randn_like(target)

            # Update EMA
            if step == 0:
                ema_entropy = entropy_noisy
            else:
                ema_entropy = update_ema(ema_entropy, entropy_noisy, beta=ema_beta)

            # Compute delta
            delta = compute_temperature_delta(ema_entropy, target, kp=kp, max_step=max_step)

            # Update temperature
            temp = update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)

            temps.append(temp.item())

        temps_tensor = torch.tensor(temps)
        metrics = compute_controller_health_metrics(temps_tensor, temp_min=0.7, temp_max=1.0)

        print(f"\nTracking Performance:")
        print(f"  Final temp: {temps[-1]:.4f}")
        print(f"  temp_sat_frac: {metrics['temp_sat_frac']:.3f}")
        print(f"  oscillation: {metrics['oscillation']:.3f}")

        # With reduced gain, should track reasonably well
        assert metrics["temp_sat_frac"] < 0.35, "Should not saturate excessively"

    def test_controller_response_to_step_change(self):
        """Controller should respond to sudden entropy change."""
        target_initial = torch.tensor([[[2.0]]])
        target_final = torch.tensor([[[3.0]]])

        kp = 0.35
        ema_beta = 0.9
        max_step = 0.0005

        temp = torch.tensor([[[1.0]]])
        ema_entropy = torch.tensor([[[2.0]]])

        temps = []

        # Step change at t=50
        for step in range(100):
            target = target_initial if step < 50 else target_final

            # Measure current entropy (simulate it being near target for simplicity)
            entropy_current = target + 0.1 * torch.randn_like(target)

            ema_entropy = update_ema(ema_entropy, entropy_current, beta=ema_beta)
            delta = compute_temperature_delta(ema_entropy, target, kp=kp, max_step=max_step)
            temp = update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)

            temps.append(temp.item())

        # Should see temperature change after step at t=50
        temp_before = np.mean(temps[40:50])
        temp_after = np.mean(temps[60:70])

        print(f"\nStep Response:")
        print(f"  Temp before: {temp_before:.4f}")
        print(f"  Temp after: {temp_after:.4f}")

        assert temp_after < temp_before, "Should decrease temp in response to higher target entropy"
