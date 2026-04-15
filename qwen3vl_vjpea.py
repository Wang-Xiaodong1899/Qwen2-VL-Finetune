import argparse
import os
import sys
import torch
import numpy as np

from transformers import AutoConfig, AutoModel, AutoProcessor, AutoVideoProcessor

from qwen_vl_utils import process_vision_info


def _load_video_for_vjepa2(video_path: str, num_frames: int) -> torch.Tensor:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_NUM_THREADS = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=TORCHCODEC_NUM_THREADS)
    total_frames = decoder.metadata.num_frames
    
    # 均匀采样 num_frames 帧
    if total_frames >= num_frames:
        # 在 [0, total_frames-1] 范围内均匀取 num_frames 个索引
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)
    else:
        # 如果视频帧数不足，则重复最后一帧（或可以选择重复采样）
        indices = np.arange(total_frames)
        # 重复最后一帧直到达到 num_frames
        indices = np.pad(indices, (0, num_frames - total_frames), constant_values=total_frames - 1)
    
    video = decoder.get_frames_at(indices=indices.tolist()).data
    return video

def _compute_vjepa_visual_embeds(
    *,
    vjepa2_model_id: str,
    video_path: str,
    device: torch.device,
    dtype: torch.dtype,
    num_frames: int,
    stride: int,
) -> torch.Tensor:
    vjepa2 = AutoModel.from_pretrained(vjepa2_model_id, torch_dtype=dtype).to(device).eval()
    vjepa2.requires_grad_(False)
    vjepa2_processor = AutoVideoProcessor.from_pretrained(vjepa2_model_id)

    with torch.inference_mode():
        video = _load_video_for_vjepa2(video_path, num_frames=num_frames)
        x = vjepa2_processor(video, return_tensors="pt")["pixel_values_videos"].to(device=device, dtype=dtype)
        feats = vjepa2.get_vision_features(x)

    return feats


def _enable_qwen3_vjepa_deepstack(model, *, vjepa2_model_id: str, checkpoint_path: str | None = None) -> None:
    from model.load_model import get_qwen_vl_generation_backbone
    from model.qwen3_vl_vjepa_deepstack import Qwen3VLTextModelWithVJEPADeepstack
    import torch.nn as nn

    backbone = get_qwen_vl_generation_backbone(model)
    if backbone.config.model_type != "qwen3_vl":
        raise ValueError(f"VJEPA deepstack only wired for qwen3_vl, got {backbone.config.model_type!r}")

    if not isinstance(backbone.language_model, Qwen3VLTextModelWithVJEPADeepstack):
        backbone.language_model.__class__ = Qwen3VLTextModelWithVJEPADeepstack
        backbone.language_model._vjepa_visual_embeds = None

    if getattr(backbone.language_model, "vjepa_zero_proj", None) is None:
        print(f"Warning: vjepa_zero_proj is None, initializing it now...")
        vjepa_cfg = AutoConfig.from_pretrained(vjepa2_model_id)
        vjepa_hidden = getattr(vjepa_cfg, "hidden_size", None)
        if vjepa_hidden is None:
            raise ValueError(f"Cannot infer VJEPA2 hidden_size from config: {type(vjepa_cfg)!r}")

        proj = nn.Linear(int(vjepa_hidden), int(backbone.language_model.config.hidden_size), bias=True)
        nn.init.zeros_(proj.weight)
        nn.init.zeros_(proj.bias)

        ref_param = next(backbone.language_model.parameters())
        backbone.language_model.vjepa_zero_proj = proj.to(device=ref_param.device, dtype=ref_param.dtype)

    if checkpoint_path is None:
        return

    try:
        from safetensors.torch import load_file
    except Exception:
        return

    import glob

    weight_key = "model.language_model.vjepa_zero_proj.weight"
    bias_key = "model.language_model.vjepa_zero_proj.bias"

    candidates: list[str] = []
    if os.path.isdir(checkpoint_path):
        candidates.append(os.path.join(checkpoint_path, "model.safetensors"))
        candidates.extend(sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors"))))
        candidates.extend(sorted(glob.glob(os.path.join(checkpoint_path, "*.safetensors"))))

    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)

        try:
            state = load_file(path, device="cpu")
        except Exception:
            continue

        if weight_key not in state or bias_key not in state:
            continue

        proj = backbone.language_model.vjepa_zero_proj
        proj.weight.data.copy_(state[weight_key].to(device=proj.weight.device, dtype=proj.weight.dtype))
        proj.bias.data.copy_(state[bias_key].to(device=proj.bias.device, dtype=proj.bias.dtype))
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/mnt/bn/wxd-video-understanding/wangxd/wm-project/Qwen2-VL-Finetune/output/normal_train/checkpoint-1500")
    parser.add_argument("--video_path", type=str, default="sample_video.mp4")
    parser.add_argument("--prompt", type=str, default="describe the video")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--vjepa2_model_id", type=str, default="/mnt/bn/wxd-video-understanding/wangxd/wm-project/Qwen2-VL-Finetune/vjepa2-vitg-fpc64-384")
    parser.add_argument("--vjepa2_num_frames", type=int, default=64)
    parser.add_argument("--vjepa2_frame_stride", type=int, default=2)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    from model.load_model import load_qwen_vl_generation_model

    compute_dtype = torch.bfloat16
    model = load_qwen_vl_generation_model(
        args.model_path,
        dtype=compute_dtype,
        attn_implementation=args.attn_implementation,
        device_map=None if args.device_map in {"none", "None", ""} else args.device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    device = next(model.parameters()).device

    vjepa_visual_embeds = None
    if args.vjepa2_model_id is not None:
        _enable_qwen3_vjepa_deepstack(
            model,
            vjepa2_model_id=args.vjepa2_model_id,
            checkpoint_path=args.model_path,
        )
        vjepa_visual_embeds = _compute_vjepa_visual_embeds(
            vjepa2_model_id=args.vjepa2_model_id,
            video_path=args.video_path,
            device=device,
            dtype=compute_dtype,
            num_frames=args.vjepa2_num_frames,
            stride=args.vjepa2_frame_stride,
        )
        print(f"vjepa_visual_embeds shape: {vjepa_visual_embeds.shape}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": args.video_path,
                    "total_pixels": 20480 * 32 * 32,
                    "min_pixels": 64 * 32 * 32,
                },
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    if isinstance(video_kwargs, list):
        video_kwargs = (video_kwargs[0] if len(video_kwargs) > 0 else None) or {}
    else:
        video_kwargs = video_kwargs or {}
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
        if len(videos) == 0:
            videos, video_metadatas = None, None
    else:
        video_metadatas = None

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    processor_kwargs = {
        "text": text,
        "return_tensors": "pt",
        "do_resize": False,
        **video_kwargs,
    }

    if images is not None:
        if isinstance(images, list) and (len(images) == 0 or images[0] is None):
            images = None
        if images is not None and getattr(model.config, "model_type", "") in {"qwen3_vl", "qwen3_vl_moe"}:
            processor_kwargs["images"] = [images]
        elif images is not None:
            processor_kwargs["images"] = images

    if videos is not None:
        if getattr(model.config, "model_type", "") in {"qwen3_vl", "qwen3_vl_moe"}:
            processor_kwargs["videos"] = [videos]
            processor_kwargs["video_metadata"] = [video_metadatas]
        else:
            processor_kwargs["videos"] = videos
            processor_kwargs["video_metadata"] = video_metadatas

    inputs = processor(**processor_kwargs)

    # Robust handling for Qwen3-VL rope index:
    # - get_rope_index iterates over image_grid_thw/video_grid_thw once per *contiguous* vision segment in mm_token_type_ids.
    # - If grids are empty or shorter than the number of segments, it raises StopIteration.
    mm_token_type_ids = inputs.get("mm_token_type_ids", None)
    if mm_token_type_ids is not None:
        attention_mask = inputs.get("attention_mask", None)

        def _count_modality_groups(token_types: torch.Tensor, mask: torch.Tensor | None, modality: int) -> int:
            if mask is not None:
                token_types = token_types[mask.bool()]
            if token_types.numel() == 0:
                return 0
            boundaries = torch.ones((token_types.numel(),), device=token_types.device, dtype=torch.bool)
            boundaries[1:] = token_types[1:] != token_types[:-1]
            return int((token_types[boundaries] == modality).sum().item())

        required_image_grids = 0
        required_video_grids = 0
        for b in range(mm_token_type_ids.shape[0]):
            m = attention_mask[b] if attention_mask is not None else None
            required_image_grids += _count_modality_groups(mm_token_type_ids[b], m, 1)
            required_video_grids += _count_modality_groups(mm_token_type_ids[b], m, 2)

        image_grid_thw = inputs.get("image_grid_thw", None)
        video_grid_thw = inputs.get("video_grid_thw", None)

        bad_image = required_image_grids > 0 and (image_grid_thw is None or image_grid_thw.numel() == 0)
        bad_video = required_video_grids > 0 and (video_grid_thw is None or video_grid_thw.numel() == 0)
        if bad_image or bad_video:
            inputs.pop("mm_token_type_ids", None)
        else:
            if image_grid_thw is not None and required_image_grids > 0:
                if int(image_grid_thw.shape[0]) != int(required_image_grids):
                    inputs.pop("mm_token_type_ids", None)

            if "mm_token_type_ids" in inputs and video_grid_thw is not None and required_video_grids > 0:
                if int(video_grid_thw.shape[0]) != int(required_video_grids):
                    temporal_sum = int(video_grid_thw[:, 0].to(torch.long).sum().item())
                    if temporal_sum == int(required_video_grids) and required_video_grids > int(video_grid_thw.shape[0]):
                        repeats = video_grid_thw[:, 0].to(torch.long)
                        hw = video_grid_thw[:, 1:].repeat_interleave(repeats, dim=0)
                        ones = torch.ones((hw.shape[0], 1), device=video_grid_thw.device, dtype=video_grid_thw.dtype)
                        inputs["video_grid_thw"] = torch.cat([ones, hw], dim=1)
                    else:
                        inputs.pop("mm_token_type_ids", None)

    inputs = inputs.to(device)
    inputs_device = inputs["input_ids"].device

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    if vjepa_visual_embeds is not None:
        gen_kwargs["vjepa_visual_embeds"] = vjepa_visual_embeds.to(device=inputs_device)

    generated_ids = model.generate(**inputs, **gen_kwargs)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])


if __name__ == "__main__":
    main()