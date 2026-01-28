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

        # normalize entropy so prompt/decode are comparable
        norm = torch.log(
            torch.tensor(float(kv_len), device=entropy_last.device)
        ).clamp(min=1.0)

        H_norm = entropy_last / norm

        # EMA smoothing
        self.ema_entropy.mul_(self.ema_beta).add_(H_norm * (1 - self.ema_beta))

        # error signal
        if self.prompt_target_entropy is not None:
            err = self.ema_entropy - self.prompt_target_entropy
        else:
            # fallback: pure sharpening when entropy is high
            err = self.ema_entropy

        # proportional control (operate in temp space)
        delta = -self.kp * err
        delta = delta.clamp(-self.max_step, self.max_step)

        self.temp.add_(delta)
        self.temp.clamp_(self.temp_min, self.temp_max)

        return self.temp
