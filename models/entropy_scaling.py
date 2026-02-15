# models/entropy_scaling.py

import torch


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
        max_step=0.05,
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
        """
        target_entropy: [Z, H, 1], normalized
        """
        self.prompt_target_entropy = target_entropy.detach()

    # ---------- update ----------

    @torch.no_grad()
    def update(self, entropy_last: torch.Tensor, kv_len: int):
        """
        entropy_last: [Z, H, 1] (last query token)
        kv_len: current KV cache length
        """
        if self.temp is None:
            self._init_state(entropy_last.shape, entropy_last.device)

        # Safe hard-disable: keep temperature fixed at init when scaling is disabled.
        if self.max_step <= 0:
            self.temp.fill_(self.temp_init)
            return self.temp

        # normalize entropy so prompt/decode are comparable
        norm = torch.log(
            torch.tensor(float(kv_len), device=entropy_last.device)
        ).clamp(min=1.0)

        H_norm = entropy_last / norm
        valid_entropy = torch.isfinite(H_norm)
        H_safe = torch.where(valid_entropy, H_norm, torch.zeros_like(H_norm))

        # EMA smoothing: update only finite lanes; keep previous value otherwise.
        ema_new = self.ema_entropy * self.ema_beta + H_safe * (1 - self.ema_beta)
        self.ema_entropy = torch.where(valid_entropy, ema_new, self.ema_entropy)

        # error signal
        if self.prompt_target_entropy is not None:
            valid_target = torch.isfinite(self.prompt_target_entropy)
            valid = valid_entropy & valid_target
            target = torch.where(valid_target, self.prompt_target_entropy, torch.zeros_like(self.prompt_target_entropy))
            err = self.ema_entropy - target
        else:
            # fallback: pure sharpening when entropy is high
            valid = valid_entropy
            err = self.ema_entropy

        # proportional control (operate in temp space)
        delta = -self.kp * err
        delta = delta.clamp(-self.max_step, self.max_step)
        delta = torch.where(valid, delta, torch.zeros_like(delta))

        self.temp.add_(delta)
        self.temp.clamp_(self.temp_min, self.temp_max)

        return self.temp
