# models/entropy_scaling.py

import torch

class EntropyTempController:
    """
    Accuracy-first entropy -> temperature controller.

    High entropy  -> decrease temperature (sharpen attention)
    Low entropy   -> relax temperature slightly

    Per-layer, per-head state
    EMA-smoothed entropy
    Entropy normalized by log(kv_len)
    clamp temp to [temp_min, temp_max]
    cap change per step
    Step-limited updates (prevents jitter)

    Works for both prefill and decode

    """

    def __init__(
        self,
        temp_init=1.0,
        temp_min=0.8, # lower bound
        temp_max=1.2, # upper bound  # or 1?
        ema_beta=0.9, # EMA smoothing
        kp=0.4,
        max_step=0.05, # step limit
        device=None,
    ):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.ema_beta = ema_beta
        self.kp = kp
        self.max_step = max_step

        self.temp = None
        self.ema_entropy = None
        self.device = device
        self.temp_init = temp_init

    def _init_state(self, shape, device):
        self.temp = torch.full(shape, self.temp_init, device=device)
        self.ema_entropy = torch.zeros(shape, device=device)

    @torch.no_grad()
    def update(self, entropy_last, kv_len):
        """
        EMA-smoothed, step-limited proportional scaling
        entropy_last: [Z, H, 1]  (last query token only)
        kv_len: int (current KV cache length)
        """
        if self.temp is None:
            self._init_state(entropy_last.shape, entropy_last.device)

        # Normalize entropy by kv length to ~[0,1]
        norm = torch.log(torch.tensor(float(kv_len), device=entropy_last.device))
        H_norm = entropy_last / norm.clamp(min=1.0)

        # EMA smoothing (0.9 old + 0.1 new)
        self.ema_entropy.mul_(self.ema_beta).add_(H_norm * (1 - self.ema_beta)) 

        # Proportional control:
        # delta = -self.kp * self.ema_entropy
        H_star = 0.45 # threshold
        # delta = -self.kp * (self.ema_entropy - H_star)
        error = self.ema_entropy - H_star
        error = torch.where(error > 0, error, torch.zeros_like(error)) # only for higher entropy -> reduce temp (sharpen attention)
        delta = -self.kp * error

        # Step limit for gradual temp update
        delta = delta.clamp(-self.max_step, self.max_step)

        self.temp.add_(delta)
        self.temp.clamp_(self.temp_min, self.temp_max)

        # if torch.rand(1).item() < 0.001:
        #     print(
        #         f"H={self.ema_entropy.mean():.3f} "
        #         f"temp={self.temp.mean():.3f}"
        #     )

        return self.temp
