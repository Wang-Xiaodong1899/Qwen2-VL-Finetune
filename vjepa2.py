from transformers import AutoVideoProcessor, AutoModel
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader

def get_video():
    vr = VideoReader("sample_video.mp4")
    # choosing some frames here, you can define more complex sampling strategy
    frame_idx = np.arange(0, 128, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video

def forward_vjepa_video(model_hf, hf_transform):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = get_video()  # T x H x W x C
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

        print(out_patch_features_hf.shape)

hf_repo = "vjepa2-vitg-fpc64-384"

model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)
model = model.cuda().eval()

forward_vjepa_video(model, processor)
