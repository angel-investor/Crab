#!/bin/bash

###########################################################
# 消融实验推理：不带 CrossAttn 的对照组
###########################################################

llama2_ckpt_path=/root/autodl-tmp/Crab/pretrain/llama2
ABLATION_CKPT_DIR=/root/autodl-tmp/Crab/results/finetune_c2_ablation/avqa_no_crossattn/checkpoint-997

echo "=================================================="
echo "   消融实验推理（不带 CrossAttn）"
echo "   ckpt_dir = $ABLATION_CKPT_DIR"
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
    --ckpt_dir $ABLATION_CKPT_DIR \
    --avqa_task True \
    --ave_task False \
    --avvp_task False \
    --arig_task False \
    --avcap_task False \
    --ms3_task False \
    --s4_task False \
    --avss_task False \
    --ref_avs_task False \
    --use_av_crossattn False \
    --mask_audio_for_ablation False \
    --test_name test \
    --device cuda:0 \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 4 \
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
