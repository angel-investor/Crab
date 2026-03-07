#!/bin/bash

###########################################################
# C2 实验：音频显式对齐模块（AVCrossAttentionFusion）微调脚本
# 策略：在原有预训练/多任务微调权重的基础上，仅额外解冻 av_fusion 层进行微调
# 耗时预估：比全量微调快很多（新增参数极少），单卡约 1-2 小时/epoch
###########################################################

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_PORT=6667
RANK=0

llama2_ckpt_path=/root/autodl-tmp/Crab/pretrain/llama2

# Training Arguments
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
GLOBAL_BATCH_SIZE=$((WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune_c2
RUN_NAME=avqa_c2_crossattn
OUTP_DIR=results

export TOKENIZERS_PARALLELISM='true'
# export ASCEND_LAUNCH_BLOCKING='1'

echo "=================================================="
echo "   C2 实验：音视频显式对齐微调"
echo "   GLOBAL_BATCH_SIZE = $GLOBAL_BATCH_SIZE"
echo "=================================================="

# -------  单卡训练（使用 python 而非 torchrun） -------
python scripts/finetune/finetune_hyperlora.py \
    --llm_name llama \
    --model_name_or_path $llama2_ckpt_path \
    --exp_desc "c2_crossattn" \
    --freeze_backbone True \
    --lora_enable True \
    --use_hyper_lora True \
    --bits 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --pretrain_ckpt_dir /root/autodl-tmp/Crab/pretrain_ckpt \
    --avqa_task True \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task False \
    --use_av_crossattn True \
    --save_modules vl_projector,al_projector,lora,av_fusion \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 8 \
    --vit_ckpt_path /root/autodl-tmp/Crab/pretrain/clip \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /root/autodl-tmp/Crab/pretrain/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --bert_ckpt_path /root/autodl-tmp/Crab/pretrain/bert \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 112 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --avs_query_num 300 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --ce_loss_weight 1.0 \
    --dice_loss_weight 0.5 \
    --bce_loss_weight 1.0 \
    --output_dir $OUTP_DIR/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.5 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 4 \
    --report_to tensorboard
