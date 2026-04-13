import json

json_path = "/mnt/bn/wxd-video-understanding/wangxd/Qwen3VL-Video-GRPO/ms-swift/data/Video-R1-260k-filter-video-mc.json"
with open(json_path, 'r') as f:
    data = json.load(f)

save_data = []
for item in data:
    id = item["problem_id"]
    video = item["videos"]
    conversations =  [
      {
        "from": "human",
        "value": f'<video>\n{item["problem"]}'
      },
      {
        "from": "gpt",
        "value": item["answer"].replace("<answer>", "").replace("</answer>", "")
      }
    ]
    save_data.append({
        "id": id,
        "video": video,
        "conversations": conversations,
    })

with open(f"Video-R1-{len(save_data)}.json", 'w') as f:
    json.dump(save_data, f, indent=4, ensure_ascii=False)
