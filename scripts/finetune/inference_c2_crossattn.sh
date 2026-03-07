#!/bin/bash

###########################################################
# C2 推理评估脚本：使用带音视频 Cross-Attention 的模型推理 AVQA 测试集
# 对应微调脚本：finetune_c2_crossattn.sh
###########################################################

llama2_ckpt_path=/root/autodl-tmp/Crab/pretrain/llama2

# C2 微调后的权重目录（与 finetune 脚本中 output_dir 对应）
C2_CKPT_DIR=results/finetune_c2/avqa_c2_crossattn

echo "=================================================="
echo "   C2 实验：音视频显式对齐推理评估"
echo "   ckpt_dir = $C2_CKPT_DIR"
echo "=================================================="

python scripts/finetune/inference_hyper_lora.py \
    --llm_name llama \
    --model_name_or_path $llama2_ckpt_path \
    --freeze_backbone True \
    --lora_enable True \
    --use_hyper_lora True \
    --use_process True \
    --bits 32 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --ckpt_dir $C2_CKPT_DIR \
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
    --mask_audio_for_ablation False \
    --test_name test \
    --device cuda:0 \
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
    --output_dir 'test'
