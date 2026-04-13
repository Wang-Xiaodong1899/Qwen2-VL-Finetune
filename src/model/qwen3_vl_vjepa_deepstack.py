import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextModel,
    Qwen3VLVisionModel,
)


class Qwen3VLTextModelWithVJEPADeepstack(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.vjepa_zero_proj = None
        self._vjepa_visual_embeds = None

    def _resample_sequence(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(0) == target_len:
            return x
        x = x.transpose(0, 1).unsqueeze(0)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x.squeeze(0).transpose(0, 1)

    def _align_vjepa_visual_embeds(
        self,
        vjepa_visual_embeds: torch.Tensor,
        visual_pos_masks: torch.Tensor,
    ) -> torch.Tensor:
        if vjepa_visual_embeds.dim() == 2:
            return vjepa_visual_embeds
        if vjepa_visual_embeds.dim() != 3:
            raise ValueError(
                f"vjepa_visual_embeds must be 2D (N_vis, D) or 3D (B, N, D), got shape={tuple(vjepa_visual_embeds.shape)}"
            )

        batch_size = vjepa_visual_embeds.size(0)
        pieces = []
        for b in range(batch_size):
            target_len = int(visual_pos_masks[b].sum().item())
            if target_len == 0:
                continue
            x = vjepa_visual_embeds[b]
            x = self._resample_sequence(x, target_len)
            pieces.append(x)

        if len(pieces) == 0:
            return vjepa_visual_embeds.new_zeros((0, vjepa_visual_embeds.size(-1)))
        return torch.cat(pieces, dim=0)

    def forward(self, *args, vjepa_visual_embeds: torch.Tensor | None = None, **kwargs):
        self._vjepa_visual_embeds = vjepa_visual_embeds
        try:
            return super().forward(*args, **kwargs)
        finally:
            self._vjepa_visual_embeds = None

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)

        hidden_states = hidden_states.clone()
        local_this = hidden_states[visual_pos_masks, :] + visual_embeds

        vjepa_visual_embeds = self._vjepa_visual_embeds
        if vjepa_visual_embeds is not None and bool(visual_pos_masks.any().item()):
            vjepa_visual_embeds = vjepa_visual_embeds.to(hidden_states.device, hidden_states.dtype)
            vjepa_visual_embeds = self._align_vjepa_visual_embeds(vjepa_visual_embeds, visual_pos_masks)

            if self.vjepa_zero_proj is None:
                raise RuntimeError(
                    "vjepa_zero_proj is not initialized. Initialize it before training (e.g., in train_sft.py) to avoid creating new modules during forward under DeepSpeed ZeRO-3."
                )
            if self.vjepa_zero_proj.in_features != vjepa_visual_embeds.size(-1):
                raise RuntimeError(
                    f"vjepa_zero_proj.in_features={self.vjepa_zero_proj.in_features} does not match vjepa_visual_embeds dim={vjepa_visual_embeds.size(-1)}"
                )
            print(f"before proj: {vjepa_visual_embeds.shape}")
            local_this = local_this + self.vjepa_zero_proj(vjepa_visual_embeds)
            print(f"after proj: {local_this.shape}")

        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states


class Qwen3VLModelWithVJEPADeepstack(Qwen3VLModel):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModelWithVJEPADeepstack._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()