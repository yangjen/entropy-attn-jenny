import torch
from typing import Optional
from models.entropy_attn_triton import attention as entropy_attention
from transformers.utils import logging
from models.entropy_scaling import EntropyTempController

logger = logging.get_logger(__name__)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def entropy_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`entropy` attention does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    logger.warning_once(f"WARNING: entropy attention backward and custom attention masking across the batch is not implemented at this time")

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # Note that it is important to check first for the shape, otherwise compile will fail with `argument 'is_causal' must be bool, not SymBool`
    if is_causal is None:
        # The last condition is for encoder (decoder) models which specify this by passing their own `is_causal` flag
        # This is mainly due to those models having mixed implementations for encoder, decoder, and encoder-decoder attns
        is_causal = query.shape[2] > 1 and attention_mask is None and getattr(module, "is_causal", True)

    # print(f"{scaling=} {is_causal=}")
    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    Z, H, N_CTX, D = query.size()

    # if hasattr(module, "past_entropy"):
    #     # calculate the temperature somehow based on the past entropy
    #     temp = torch.ones(Z, H, N_CTX, device=query.device, dtype=query.dtype)
    # else:
    #     temp = torch.ones(Z, H, N_CTX, device=query.device, dtype=query.dtype)
    """
    update temp using last token entropy, keep per-head state
    Maintain a per-layer, per-head scalar temperature state (e.g., [Z, H, 1])

    """
    # ---------- entropy-conditioned temperature (single controller) ----------
    if not hasattr(module, "_entropy_temp_controller"):
        module._entropy_temp_controller = EntropyTempController(
            temp_init=1.0,
            temp_min=0.7,
            temp_max=1.0,
            ema_beta=0.9,
            kp=0.35,
            max_step=0.05,
        )

    controller = module._entropy_temp_controller

    # initialize controller state once
    if controller.temp is None:
        controller._init_state((Z, H, 1), query.device)

    temp = controller.temp.expand(Z, H, N_CTX)

    # ---------- attention ----------
    attn_output, attn_entropy = entropy_attention(
        query, key, value,
        is_causal, scaling,
        temp
    )

    # ---------- prompt entropy reference (prefill only) ----------
    if N_CTX > 1 and controller.prompt_target_entropy is None:
        kv_len = key.shape[2]

        H_norm = attn_entropy / torch.log(
            torch.tensor(float(kv_len), device=attn_entropy.device)
        ).clamp(min=1.0)

        K = min(256, H_norm.shape[-1])
        tail = H_norm[:, :, -K:]              # [Z, H, K]

        prompt_target = tail.mean(dim=-1, keepdim=True)
        controller.set_prompt_target(prompt_target)

    # ---------- decode-time entropy feedback ----------
    if N_CTX == 1:
        entropy_last = attn_entropy[:, :, -1:].detach()
        kv_len = key.shape[2]

        controller.update(entropy_last, kv_len)

        module.past_entropy = entropy_last
        module.past_temp = controller.temp.detach()

    # ---- sanity check (temporary) ----
    if N_CTX == 1 and controller.prompt_target_entropy is not None and torch.rand(1).item() < 0.01:
        layer_idx = getattr(module, "layer_idx", "?")
        print(
            f"[layer {layer_idx}] "
            f"H*={controller.prompt_target_entropy.mean().item():.3f} "
            f"H={controller.ema_entropy.mean().item():.3f} "
            f"T={controller.temp.mean().item():.3f}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None
