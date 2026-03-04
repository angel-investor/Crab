#!/bin/bash

# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_PORT=6666
RANK=0

llama2_ckpt_path=/root/autodl-tmp/Crab/pretrain/llama2
qwen2_ckpt_path=/root/autodl-tmp/Crab/pretrain/qwen2
dockerdata_llama2_ckpt_path=/root/autodl-tmp/Crab/pretrain/llama2

# Training Arguments
LOCAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=1
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS
# 16*8*4
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=finetune
RUN_NAME=ms3-s4
OUTP_DIR=results
# export CUDA_VISIBLE_DEVICES='0'
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'
# export NCCL_P2P_DISABLE=NVL
# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
avs_ckpt_dir=/root/autodl-tmp/Crab/avs_ckpt
avss_ckpt_dir=/root/autodl-tmp/Crab/avs_ckpt

python scripts/quick_start.py \
    --llm_name llama \
    --model_name_or_path $dockerdata_llama2_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --use_hyper_lora True \
    --use_process True \
    --bits 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 False \
    --tf32 False \
    --fp16 False \
    --ckpt_dir /root/autodl-tmp/Crab/ckpt/ \
    --avs_ckpt_dir $avs_ckpt_dir \
    --avqa_task False \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task True \
    --test_name test_s \
    --device cuda:0 \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path /root/autodl-tmp/Crab/pretrain/clip \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path /root/autodl-tmp/Crab/pretrain/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
    --bert_ckpt_path /root/autodl-tmp/Crab/pretrain/bert \
    --audio_query_token_nums 32 \
    --seg_branch True \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 112 \
    --image_scale_nums 1 \
    --token_nums_per_scale 5 \
    --avs_query_num 300 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --output_dir 'test'


