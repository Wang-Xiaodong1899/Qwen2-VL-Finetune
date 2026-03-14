#!/bin/bash

export WANDB_PROJECT=Qwen2.5-VL-7B-Game-SFT
export WANDB_NAME=Qwen2.5-VL-7B-Game-SFT-game-9k-epoch-4-f64-spatial128
# export WANDB_MODE=offlinew

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-7B-Instruct/"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=16
BATCH_PER_DEVICE=2
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# NOTE Direct Training with QA samples

# remember use Zero-3 to training

deepspeed src/train/train_game.py \
    --use_liger False \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /root/Open-R1-Video-V1/game/all_game_cls_dir_merged-7B-pred-s0-e12100-train.json \
    --image_folder xxx \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME} \
    --num_train_epochs 4 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((64 * 28 * 28)) \
    --image_max_pixels $((128 * 28 * 28)) \
    --video_min_pixels $((64 * 28 * 28)) \
    --video_max_pixels $((128 * 28 * 28)) \
    --video_max_frames 64 \
    --video_min_frames 16 \
    --fps 1.0 \
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
    --save_steps 500 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --max_grad_norm 0.5 \