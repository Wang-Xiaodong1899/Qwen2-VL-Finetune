#!/bin/bash

export WANDB_PROJECT=Qwen2-VL-Ins-7B-Video-DPO
export WANDB_NAME=Qwen2-VL-Ins-7B_sdf_17k_DPO-5e-7


# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="/mnt/bn/multimodal-datasets-hl/wangxd/models/Qwen2.5-VL-7B-Instruct/"
MODEL_NAME="/mnt/bn/ws-candy-hl-62827-yz89lqpbo2/models/Qwen2-VL-7B-Instruct"


GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_dpo.py \
    --dpo_loss "sigmoid" \
    --precompute_ref_log_probs False \
    --beta 0.1 \
    --use_liger False \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /root/Open-R1-Video-V1/Qwen2-VL-Finetune/scripts/sft_dpo_17k_add_videotok.json \
    --image_folder /mnt/bn/multimodal-datasets-hl/wangxd/data/shareVideoGPTV/dpo_train_data \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /mnt/bn/multimodal-datasets-hl/wangxd/ckpt/${WANDB_PROJECT}/${WANDB_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((128 * 28 * 28)) \
    --image_max_pixels $((196 * 28 * 28)) \
    --video_min_pixels $((128 * 28 * 28)) \
    --video_max_pixels $((196 * 28 * 28)) \
    --video_max_frames 32 \
    --video_min_frames 16 \
    --learning_rate 5e-7 \
    --merger_lr 5e-7 \
    --vision_lr 5e-7 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --dataloader_num_workers 4