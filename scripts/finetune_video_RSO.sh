#!/bin/bash

export WANDB_PROJECT=Qwen3-VL-8B-Video-RSO

export FPS_MAX_FRAMES=32
export WANDB_NAME=hound-17k-first-round-1e-6-ep1-f${FPS_MAX_FRAMES}


# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

MODEL_NAME="/mnt/bn/wxd-video-understanding/wangxd/models/Qwen3-VL-8B-Instruct/"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If your dataset is mixed with images and videos, you need to use zero2.
# If you want to set the min pixels and max pixels for Qwen3-VL, You should set as (N * 32 * 32)

deepspeed src/train/train_sft.py \
    --use_liger False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /mnt/bn/wxd-video-understanding/wangxd/Qwen3VL-Video-GRPO/Qwen2-VL-Finetune/data/sft_dpo_17k_add_videotok_win_4RSO.json \
    --image_folder /mnt/bn/wxd-video-understanding/wangxd/data/shareVideoGPTV/dpo_train_data \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /mnt/bn/wxd-video-understanding/wangxd/Qwen3VL-Video-GRPO/Qwen2-VL-Finetune/ckpt/${WANDB_PROJECT}/${WANDB_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((360 * 420)) \
    --learning_rate 1e-6 \
    --merger_lr 1e-6 \
    --vision_lr 1e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 250 \
    --save_total_limit 10 \
    --dataloader_num_workers 4