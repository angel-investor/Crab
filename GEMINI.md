# Crab 项目规则

## 项目简介

**Crab** 是 GeWu-Lab 发表于 **CVPR 2025** 的工作，全称为 **"A Unified Audio-Visual Scene Understanding Model with Explicit Cooperation"**（融合显式协作的统一音视频场景理解模型）。该项目基于大语言模型（LLaMA-2-7B-Chat），通过引入视觉编码器和音频编码器，构建了一个能够统一处理多种音视频理解任务的多模态大模型。

### 支持的任务

| 任务类型 | 任务名称 | 说明 |
|---|---|---|
| **时序定位** | AVE | 音视频事件时序定位 |
| **时序定位** | AVVP | 音视频视频解析（细粒度时序定位）|
| **时空推理** | MUSIC-AVQA | 音视频问答（空间 + 时序）|
| **空间定位** | ARIG | 音频参考图像定位 |
| **像素级分割** | AVS (S4/MS3) | 音频驱动的视频分割 |
| **像素级分割** | Ref-AVS | 文本引导的音视频分割 |
| **像素级分割** | AVSS | 音视频语义分割 |
| **描述生成** | AV-Cap | 音视频描述生成 |

---

## 整体架构

Crab 采用 **编码器 → 投影器 → 大语言模型** 的经典多模态架构，并在此基础上增加了分割解码分支和 Hyper-LoRA 任务自适应机制。

```
视频帧 ──→ VisualEncoder (CLIP ViT-L/14) ──→ VLProjector (Q-Former + MLP) ──┐
                                                                              ├──→ LLM (LLaMA-2-7B-Chat / Qwen) ──→ 文本输出
音频流 ──→ AudioEncoder (BEATs iter3+)   ──→ ALProjector (Q-Former + MLP) ──┘
                                                                              │
图像帧 ──→ VisualEncoder (多尺度特征)    ─────────────────────────────────→ SegModule (SAM-style 解码器) ──→ 分割 Mask
```

### 核心模块说明

#### 1. 视觉编码器（`models/multimodal_encoder.py` → `VisualEncoder`）
- 使用 **OpenAI CLIP ViT-Large/14**（冻结权重）
- 从第 14、22、23 层提取多尺度特征（`select_layer_list`）
- 支持图像帧和视频帧的统一编码

#### 2. 音频编码器（`models/multimodal_encoder.py` → `AudioEncoder`）
- 使用微软的 **BEATs iter3+（AS2M 微调版）**（冻结权重）
- 将音频 Mel 频谱图转为音频 Token 序列

#### 3. 多模态投影器（`VLProjector` / `ALProjector`）
- 均采用 **Q-Former（BERT-based）+ MLP** 的两阶段投影结构
- Q-Former 将变长视觉/音频特征压缩为固定长度 Token（默认 32 个）
- MLP 将特征对齐到 LLM 的隐藏维度（4096）

#### 4. 分割模块（`models/multimodal_encoder.py` → `SegModule`）
- 参考 SAM（Segment Anything）架构设计
- 包含 `PromptEncoder`、`MaskDecoder`、`Transformer`
- 利用 LLM 输出的特殊 `<mask>` Token 作为分割提示
- 支持多尺度图像特征融合（`image_scale_nums=2`）

#### 5. 统一架构基类（`models/unified_arch.py`）
- `UnifiedMetaModel`：注册并初始化所有多模态模块
- `UnifiedMetaForCausalLM`：处理多模态输入拼接、Token 替换、前向传播
- `prepare_multimodal_inputs()`：将视觉/音频 Token 注入 LLM 输入序列的核心方法

#### 6. LLM 骨干网络
- 主要支持 **LLaMA-2-7B-Chat**（`models/unified_llama.py`）
- 也支持 **Qwen**（`models/unified_qwen.py`）
- 骨干网络在训练中默认冻结

#### 7. Hyper-LoRA 参数高效微调（`peft_hyper/`）
- 基于 LoRA 思想，针对不同任务动态生成 LoRA 权重（Hyper-LoRA）
- 在微调阶段统一训练所有任务，避免多任务冲突
- 实现文件：`peft_hyper/peft_model.py`，`peft_hyper/tuners/`

---

## 目录结构

```
Crab/
├── models/                  # 模型定义
│   ├── unified_arch.py      # 统一架构基类（核心入口）
│   ├── multimodal_encoder.py# 视觉/音频编码器、投影器、分割模块
│   ├── unified_llama.py     # 基于 LLaMA 的统一模型
│   ├── unified_qwen.py      # 基于 Qwen 的统一模型
│   ├── modeling_llama.py    # 改造后的 LLaMA 模型实现
│   ├── Qformer.py           # Q-Former 实现
│   ├── mask_decoder.py      # SAM 风格 Mask 解码器
│   ├── prompt_encoder.py    # SAM 风格 Prompt 编码器
│   ├── beats/               # BEATs 音频编码器
│   └── taming_transformer/  # VQGAN（可选生成模块）
├── dataset/                 # 数据集加载
│   ├── unified_dataset.py   # 统一多任务数据集（微调核心）
│   ├── pretrain_dataset.py  # 预训练数据集
│   ├── quick_start_dataset.py # 快速推理数据集
│   ├── AVQA.py / MS3.py     # 特定任务数据处理
│   └── video_processor.py   # 视频帧抽取与预处理
├── configs/
│   └── unified_config.py    # 所有训练/推理参数的 dataclass 定义
├── peft_hyper/              # Hyper-LoRA 实现（改造自 peft 库）
├── scripts/
│   ├── pretrain/            # 视觉/音频/分割预训练脚本
│   ├── finetune/            # 多任务微调 & 推理脚本
│   ├── quick_start.py       # 快速推理入口
│   └── quick_start.sh
├── utils/                   # 工具函数
│   ├── mm_utils.py          # 多模态数据预处理工具
│   ├── constants.py         # Token 常量定义
│   ├── avvp_eval_metrics.py # AVVP 评测指标
│   └── deepspeed_utils.py   # DeepSpeed 分布式训练工具
└── deepspeed/               # DeepSpeed 配置文件
```

---

## 训练流程

1. **预训练**（三路并行）
   - 视觉预训练：`scripts/pretrain/pretrain_visual.sh`（数据：Video-LLaVA）
   - 音频预训练：`scripts/pretrain/pretrain_audio.sh`（数据：AudioCaps）
   - 分割预训练：`scripts/pretrain/pretrain_seg.sh`（数据：LVIS）

2. **多任务联合微调**
   - 数据集：自建 **AVUIE** 数据集（整合 AVE、AVVP、MUSIC-AVQA、AVS、Ref-AVS 等）
   - 脚本：`scripts/finetune/finetune_hyperlora.sh`
   - 使用 Hyper-LoRA 统一训练所有任务

3. **AVS 任务专项微调**
   - 在多任务微调权重基础上继续微调分割分支
   - 脚本：`scripts/finetune/finetune_hyper_lora_avs.sh`

4. **推理**
   - 批量推理：`scripts/finetune/inference_hyper_lora.sh`
   - 快速体验：`scripts/quick_start.sh`

---

## 代码规范
- 使用 Python 编写，遵循 PEP8 代码风格
- 所有函数和类必须添加 docstring
- 优先使用 PyTorch 框架的 API

## 回答规范
- 请始终用中文回答关于本项目的问题
- 涉及代码修改时，说明修改的原因和影响
