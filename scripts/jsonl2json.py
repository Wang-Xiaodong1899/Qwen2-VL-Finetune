import json

def jsonl_to_json_with_video_tag(jsonl_path, json_path):
    # 读取JSONL文件并处理
    items = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # 在prompt头部添加"<video>\n"
            if "prompt" in item:
                item["prompt"] = "<video>\n" + item["prompt"]
            items.append(item)
    
    # 写入JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=4)

# 使用示例
# jsonl_file = "/mnt/bn/multimodal-datasets-hl/wangxd/data/shareVideoGPTV/sft_dpo_17k.jsonl"  # 输入的JSONL文件路径
# json_file = "/root/Open-R1-Video-V1/Qwen2-VL-Finetune/scripts/sft_dpo_17k_add_videotok.json"   # 输出的JSON文件路径
# jsonl_to_json_with_video_tag(jsonl_file, json_file)
jsonl_file = "/root/Open-R1-Video-V1/data/qwen-debate-hound-17k/qwen-hound-17k-0520_xxx_merge-rej.jsonl"  # 输入的JSONL文件路径
json_file = "/root/Open-R1-Video-V1/data/qwen-debate-hound-17k/qwen-hound-17k-0520_xxx_merge-rej-add_videotok-8k.json"   # 输出的JSON文件路径
jsonl_to_json_with_video_tag(jsonl_file, json_file)