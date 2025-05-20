#!/bin/bash

export WANDB_PROJECT=Qwen2-VL-7B-Video-SFT
export WANDB_NAME=llava-178k-youtube-QA-DT-f64-fps1-MAX196-epoch-2
# export WANDB_MODE=offlinew

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/models/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=32
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# NOTE Direct Training with QA samples

# remember use Zero-3 to training

deepspeed src/training/train.py \
    --use_liger False \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /root/Open-R1-Video-V1/data/LLaVA-Video-2_3_m_youtube_mc-qwen_filter_1_QA.json \
    --image_folder /mnt/bn/multimodal-datasets-hl/wangxd/data/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/ \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME} \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --video_max_pixels $((360 * 420)) \
    --fps 1.0 \
    --learning_rate 5e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --max_grad_norm 0.5 \