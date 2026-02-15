# models/entropy_scaling.py

import torch
from models.entropy_ops import (
    normalize_entropy,
    update_ema,
    compute_temperature_delta,
    update_temperature,
)


class EntropyTempController:
    """
    Prompt-referenced entropy -> attention temperature controller.

    Goal:
      Keep decode-time attention entropy close to prompt-derived entropy.

    Behavior:
      entropy_decode > entropy_prompt  -> decrease temp (sharpen)
      entropy_decode < entropy_prompt  -> increase temp (relax)

    All operations are:
      - per-layer
      - per-head
      - bounded and EMA-smoothed
    """

    def __init__(
        self,
        temp_init=1.0,
        temp_min=0.7,
        temp_max=1.0,
        ema_beta=0.9,
        kp=0.35,
        max_step=0.0005,
    ):
        self.temp_min = temp_min
        self.temp_max = temp_max
        self.ema_beta = ema_beta
        self.kp = kp
        self.max_step = max_step

        self.temp = None                    # [Z, H, 1]
        self.ema_entropy = None             # [Z, H, 1]
        self.prompt_target_entropy = None   # [Z, H, 1]

        self.temp_init = temp_init

    # ---------- initialization ----------

    def _init_state(self, shape, device):
        self.temp = torch.full(shape, self.temp_init, device=device)
        self.ema_entropy = torch.zeros(shape, device=device)

    def set_prompt_target(self, target_entropy: torch.Tensor):
        """Set target entropy from prompt phase.

        Args:
            target_entropy: [Z, H, 1] - Normalized target entropy
        """
        self.prompt_target_entropy = target_entropy.detach()

    # ---------- update ----------

    @torch.no_grad()
    def update(self, entropy_last: torch.Tensor, kv_len: int):
        """Update temperature based on entropy error.

        Args:
            entropy_last: [Z, H, 1] - Raw entropy from last query token
            kv_len: Current KV cache length

        Returns:
            Updated temperature [Z, H, 1]
        """
        if self.temp is None:
            self._init_state(entropy_last.shape, entropy_last.device)

        # Safe hard-disable: keep temperature fixed at init when scaling is disabled.
        if self.max_step <= 0:
            self.temp.fill_(self.temp_init)
            return self.temp

        # Normalize entropy so prompt/decode are comparable
        H_norm = normalize_entropy(entropy_last, kv_len)
        valid_entropy = torch.isfinite(H_norm)
        H_safe = torch.where(valid_entropy, H_norm, torch.zeros_like(H_norm))

        # EMA smoothing: update only finite lanes; keep previous value otherwise
        self.ema_entropy = update_ema(
            self.ema_entropy,
            H_safe,
            beta=self.ema_beta,
            valid_mask=valid_entropy
        )

        # Compute temperature delta based on error
        if self.prompt_target_entropy is not None:
            # Proportional control toward target
            valid_target = torch.isfinite(self.prompt_target_entropy)
            valid = valid_entropy & valid_target

            # Safe target (replace invalid with zeros)
            target_safe = torch.where(
                valid_target,
                self.prompt_target_entropy,
                torch.zeros_like(self.prompt_target_entropy)
            )

            delta = compute_temperature_delta(
                self.ema_entropy,
                target_safe,
                kp=self.kp,
                max_step=self.max_step,
                allow_increase=False  # Only sharpen, never relax
            )
        else:
            # Fallback: pure sharpening when entropy is high
            # Treat current entropy as error (implicitly target=0)
            valid = valid_entropy
            delta = compute_temperature_delta(
                self.ema_entropy,
                torch.zeros_like(self.ema_entropy),
                kp=self.kp,
                max_step=self.max_step,
                allow_increase=False
            )

        # Zero out delta for invalid positions
        delta = torch.where(valid, delta, torch.zeros_like(delta))

        # Update temperature
        self.temp = update_temperature(
            self.temp,
            delta,
            temp_min=self.temp_min,
            temp_max=self.temp_max
        )

        return self.temp
