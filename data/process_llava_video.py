import json
import os
from tqdm import tqdm

# json_path = "2_3_m_youtube_v0_1_cap_processed.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)

json_path = "2_3_m_youtube_oe_v0_1_qa_processed.json"
with open(json_path, 'r') as f:
    data1 = json.load(f)

data2 = data1

video_root = "/mnt/bn/wxd-video-understanding/wangxd/data/LLaVA-Video-178K/2_3_m_youtube_v0_1"
save_data = []
for item in tqdm(data2):
    video = item["video"]
    video = os.path.join(video_root, video)
    if os.path.exists(video):
        item["video"] = video
        item["conversations"][0]["value"] = item["conversations"][0]["value"].replace("<image>", "<video>")
        save_data.append(item)

with open(f"LLaVA-Video-178k-2_3_m_youtube_v0_1-oe-{len(save_data)}.json", 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)