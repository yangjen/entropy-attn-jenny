"""Pure functions for entropy-based temperature control.

This module contains extracted, testable functions that implement the core
research logic for entropy-aware attention temperature scaling.

All functions are stateless and can be tested in isolation with synthetic data.
"""

import torch
from torch import Tensor
from typing import Optional


def normalize_entropy(entropy: Tensor, kv_len: int) -> Tensor:
    """Normalize entropy by log(sequence_length) to make comparable across lengths.

    The maximum entropy for a uniform distribution over N items is log(N).
    By dividing by log(kv_len), we normalize entropy to approximately [0, 1].

    Args:
        entropy: Raw entropy values, any shape
        kv_len: KV cache length (sequence length)

    Returns:
        Normalized entropy in approximately [0, 1] range

    Examples:
        >>> entropy = torch.tensor([2.0, 3.0])
        >>> normalize_entropy(entropy, kv_len=100)
        tensor([0.4343, 0.6515])
    """
    norm = torch.log(torch.tensor(float(kv_len), device=entropy.device))
    norm = norm.clamp(min=1.0)  # Avoid division by zero for kv_len=1
    return entropy / norm


def compute_entropy_from_attention_weights(attn_weights: Tensor, eps: float = 1e-9) -> Tensor:
    """Compute Shannon entropy: H = -sum(p * log(p))

    Args:
        attn_weights: [..., seq_len] - Normalized attention weights (sum to 1 over last dim)
        eps: Small constant for numerical stability

    Returns:
        entropy: [...] - Entropy per attention distribution (nats, using natural log)

    Examples:
        >>> # Uniform distribution has high entropy
        >>> uniform = torch.ones(10) / 10
        >>> compute_entropy_from_attention_weights(uniform)
        tensor(2.3026)  # ≈ log(10)

        >>> # Peaked distribution has low entropy
        >>> peaked = torch.zeros(10)
        >>> peaked[0] = 1.0
        >>> compute_entropy_from_attention_weights(peaked)
        tensor(0.)
    """
    return -(attn_weights * torch.log(attn_weights + eps)).sum(dim=-1)


def compute_target_from_tail(
    entropy_normalized: Tensor,
    tail_length: int = 256,
    trim_ratio: float = 0.0
) -> Tensor:
    """Compute target entropy from tail of sequence using trimmed mean.

    This extracts a reference entropy value from the end of the prompt sequence
    (the "tail"). A trimmed mean is used to make it robust to outliers.

    Args:
        entropy_normalized: [Z, H, seq_len] - Normalized entropy sequence
        tail_length: Number of tokens to use from end (default: 256)
        trim_ratio: Fraction to trim from each end, in [0, 0.49] (default: 0)
                   e.g., 0.1 means trim 10% from low end and 10% from high end

    Returns:
        target_entropy: [Z, H, 1] - Target for decode phase

    Examples:
        >>> H = torch.randn(1, 32, 1000)  # 1000 token prompt, 32 heads
        >>> target = compute_target_from_tail(H, tail_length=256, trim_ratio=0.1)
        >>> target.shape
        torch.Size([1, 32, 1])

    Notes:
        - If trim_ratio=0, returns simple mean of tail
        - If trim_ratio>0, sorts tail and removes outliers before averaging
        - If sequence shorter than tail_length, uses full sequence
    """
    # Clamp trim_ratio to valid range
    trim_ratio = max(0.0, min(0.49, trim_ratio))

    # Extract tail (last K tokens)
    K = min(tail_length, entropy_normalized.shape[-1])
    tail = entropy_normalized[:, :, -K:]

    # Compute trimmed mean
    trim_n = int(K * trim_ratio)

    if trim_n > 0 and (2 * trim_n) < K:
        # Sort and remove top/bottom trim_ratio
        tail_sorted = torch.sort(tail, dim=-1).values
        tail_core = tail_sorted[:, :, trim_n:(K - trim_n)]
        return tail_core.mean(dim=-1, keepdim=True)
    else:
        # No trimming (trim_ratio=0 or K too small)
        return tail.mean(dim=-1, keepdim=True)


def update_ema(
    ema_prev: Tensor,
    value_new: Tensor,
    beta: float,
    valid_mask: Optional[Tensor] = None
) -> Tensor:
    """Exponential moving average update.

    EMA provides smoothing over time:
        EMA_new = beta * EMA_old + (1 - beta) * value_new

    Higher beta → more smoothing (slower response)
    Lower beta → less smoothing (faster response)

    Args:
        ema_prev: Previous EMA value
        value_new: New observation
        beta: Smoothing factor in [0, 1]
              - beta=0: No memory (EMA = value_new)
              - beta=0.9: Strong smoothing (typical)
              - beta=0.99: Very strong smoothing
        valid_mask: Optional [bool] mask for which values to update.
                   Where False, keeps ema_prev unchanged.

    Returns:
        ema_updated: New EMA value

    Examples:
        >>> ema = torch.tensor([1.0])
        >>> update_ema(ema, torch.tensor([2.0]), beta=0.9)
        tensor([1.1000])  # 0.9 * 1.0 + 0.1 * 2.0

        >>> # With mask
        >>> ema = torch.tensor([1.0, 2.0])
        >>> new = torch.tensor([3.0, 4.0])
        >>> mask = torch.tensor([True, False])
        >>> update_ema(ema, new, beta=0.5, valid_mask=mask)
        tensor([2.0000, 2.0000])  # Only first updated
    """
    ema_new = ema_prev * beta + value_new * (1 - beta)

    if valid_mask is not None:
        return torch.where(valid_mask, ema_new, ema_prev)

    return ema_new


def compute_temperature_delta(
    entropy_current: Tensor,
    entropy_target: Tensor,
    kp: float,
    max_step: float,
    allow_increase: bool = False
) -> Tensor:
    """Compute temperature adjustment using proportional control.

    This is the CORE control law that implements the research hypothesis:
    adjusting temperature based on entropy error.

    Control Law:
        error = H_current - H_target
        delta_T = -kp * error     (negative feedback)

    Behavior:
        - If H_current > H_target: error > 0 → delta_T < 0 → decrease temp → sharpen attention
        - If H_current < H_target: error < 0 → delta_T > 0 → increase temp → broaden attention

    Args:
        entropy_current: [Z, H, 1] - Current (EMA-smoothed) entropy
        entropy_target: [Z, H, 1] - Target entropy from prompt
        kp: Proportional gain (typical: 0.1-0.5)
            - Higher kp → stronger/faster response
            - Lower kp → gentler/slower response
        max_step: Maximum absolute temperature change per step
                 Limits how fast temperature can change
        allow_increase: If False, only decrease temperature (sharpen only mode)
                       If True, allow bidirectional control

    Returns:
        delta_temp: [Z, H, 1] - Temperature change in [-max_step, +max_step]

    Examples:
        >>> # High entropy → sharpen (decrease temp)
        >>> current = torch.tensor([[[3.0]]])
        >>> target = torch.tensor([[[2.0]]])
        >>> delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05)
        >>> delta
        tensor([[[-0.0500]]])  # Negative (sharpening), saturated at -max_step

        >>> # Low entropy → relax (increase temp) if allowed
        >>> current = torch.tensor([[[1.0]]])
        >>> target = torch.tensor([[[2.0]]])
        >>> delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05, allow_increase=True)
        >>> delta
        tensor([[[0.0500]]])  # Positive (relaxing), saturated at +max_step

        >>> # With allow_increase=False (default), no relaxing
        >>> delta = compute_temperature_delta(current, target, kp=0.35, max_step=0.05, allow_increase=False)
        >>> delta
        tensor([[[0.]]])  # Zero (no change when H < target and allow_increase=False)

    Notes:
        - Default behavior (allow_increase=False) only prevents attention collapse (high entropy)
        - This asymmetry may be intentional design choice or a bug to investigate
    """
    err = entropy_current - entropy_target

    if not allow_increase:
        # Only correct high entropy (only sharpen, never relax)
        # This means: if current < target, error is negative → clamp to 0 → no action
        err = torch.clamp(err, min=0.0)

    # Proportional control with negative feedback
    delta = -kp * err

    # Limit rate of change
    delta = delta.clamp(-max_step, max_step)

    return delta


def update_temperature(
    temp_current: Tensor,
    delta: Tensor,
    temp_min: float = 0.7,
    temp_max: float = 1.0
) -> Tensor:
    """Update temperature with delta and clamp to valid range.

    Args:
        temp_current: Current temperature
        delta: Change in temperature (from compute_temperature_delta)
        temp_min: Lower bound (default: 0.7)
        temp_max: Upper bound (default: 1.0)

    Returns:
        temp_new: Updated and clamped temperature

    Examples:
        >>> temp = torch.tensor([[[0.9]]])
        >>> delta = torch.tensor([[[-0.05]]])
        >>> update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)
        tensor([[[0.8500]]])

        >>> # Clamping at lower bound
        >>> temp = torch.tensor([[[0.72]]])
        >>> delta = torch.tensor([[[-0.05]]])
        >>> update_temperature(temp, delta, temp_min=0.7, temp_max=1.0)
        tensor([[[0.7000]]])
    """
    temp_new = temp_current + delta
    return temp_new.clamp(temp_min, temp_max)


def compute_controller_health_metrics(temps: Tensor, temp_min: float, temp_max: float):
    """Compute diagnostic metrics for controller behavior.

    These metrics help identify if the controller is working properly or
    if it's saturating/oscillating excessively.

    Args:
        temps: [num_steps] - Temperature trajectory over time
        temp_min: Lower temperature bound
        temp_max: Upper temperature bound

    Returns:
        Dictionary with metrics:
        - temp_range: Max - min temperature (should use significant portion of [temp_min, temp_max])
        - temp_sat_frac: Fraction of time at bounds (should be < 0.15)
        - mean_abs_dT: Average step size (should be << max_step most of the time)
        - max_abs_dT: Largest step (will equal max_step if saturating)
        - oscillation: Fraction of direction changes (should be < 0.25)

    Ideal Values:
        - temp_range: > 0.15 (using >50% of available range)
        - temp_sat_frac: < 0.15 (at bounds <15% of time)
        - oscillation: < 0.25 (changing direction <25% of time)

    Examples:
        >>> temps = torch.linspace(0.7, 1.0, 100)  # Smooth trajectory
        >>> metrics = compute_controller_health_metrics(temps, 0.7, 1.0)
        >>> metrics['oscillation']
        0.0  # No oscillation
    """
    temps_np = temps.detach().cpu().numpy()
    dT = temps_np[1:] - temps_np[:-1]

    return {
        "temp_range": float(temps_np.max() - temps_np.min()),
        "temp_sat_frac": float(
            ((temps_np <= temp_min + 1e-4) | (temps_np >= temp_max - 1e-4)).mean()
        ),
        "mean_abs_dT": float(abs(dT).mean()) if len(dT) > 0 else 0.0,
        "max_abs_dT": float(abs(dT).max()) if len(dT) > 0 else 0.0,
        "oscillation": float((dT[1:] * dT[:-1] < 0).mean()) if len(dT) > 1 else 0.0,
    }
