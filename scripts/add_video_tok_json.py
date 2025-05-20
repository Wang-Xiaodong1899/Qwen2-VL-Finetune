import json

# read json
with open("/root/Open-R1-Video-V1/Qwen2-VL-Finetune/scripts/qwen-hound-17k-0518.json", "r") as f:
    data = json.load(f)


# add <video>/n to data['prompt']
for item in data:
    item['prompt'] = "<video>\n" + item['prompt']

# save json
with open("/root/Open-R1-Video-V1/Qwen2-VL-Finetune/scripts/qwen-hound-17k-0518-video.json", "w") as f:
    json.dump(data, f, indent=4)
