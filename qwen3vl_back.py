import argparse
import json
import os
import sys
import torch

import json
import os
import math
from tqdm import tqdm
# from decord import VideoReader, cpu

from transformers import AutoConfig

# import cv2
import base64
import openai

from PIL import Image

import numpy as np
import re

MAX_IMAGE_LENGTH = 64
from qwen_vl_utils import process_vision_info
import torch
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration

model_path = "/mnt/bn/wxd-video-understanding/wangxd/models/Qwen3-VL-8B-Instruct"

from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # 张量并行相关参数
            # low_cpu_mem_usage=True,
            # max_memory={i: '80GB' for i in range(8)},  # 根据你的GPU内存调整
        )
processor = AutoProcessor.from_pretrained(model_path)

video_path = "sample_video.mp4"

full_prompt = "describe the video"

messages = [
    {
        "role": "user",
        "content": [
            {
            "type": "video",
            "video": video_path, "total_pixels": 20480 * 32 * 32, "min_pixels": 64 * 32 * 32
            },
            {"type": "text", "text": full_prompt},
        ],
    }
]

images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)
else:
    video_metadatas = None

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, use_cache=True)

print(f'video token length: {processor.decode(generated_ids[0]).count("video_pad")}')

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
outputs_1 = output_text[0]
print(f"response: {outputs_1}")