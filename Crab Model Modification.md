# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

现在通读整个项目的架构，给我在GEMINI.md文件中加入项目的整体介绍和整体架构

*Listed directory [Crab](file:///Users/youcheng/PycharmProjects/Crab) *

*Viewed [README.md](file:///Users/youcheng/PycharmProjects/Crab/README.md) *

*Viewed [GEMINI.md](file:///Users/youcheng/PycharmProjects/Crab/GEMINI.md) *

*Listed directory [models](file:///Users/youcheng/PycharmProjects/Crab/models) *

*Listed directory [dataset](file:///Users/youcheng/PycharmProjects/Crab/dataset) *

*Listed directory [scripts](file:///Users/youcheng/PycharmProjects/Crab/scripts) *

*Listed directory [utils](file:///Users/youcheng/PycharmProjects/Crab/utils) *

*Listed directory [configs](file:///Users/youcheng/PycharmProjects/Crab/configs) *

*Listed directory [peft_hyper](file:///Users/youcheng/PycharmProjects/Crab/peft_hyper) *

*Analyzed [unified_arch.py](file:///Users/youcheng/PycharmProjects/Crab/models/unified_arch.py) *

*Analyzed [multimodal_encoder.py](file:///Users/youcheng/PycharmProjects/Crab/models/multimodal_encoder.py) *

*Listed directory [finetune](file:///Users/youcheng/PycharmProjects/Crab/scripts/finetune) *

*Listed directory [pretrain](file:///Users/youcheng/PycharmProjects/Crab/scripts/pretrain) *

*Analyzed [unified_config.py](file:///Users/youcheng/PycharmProjects/Crab/configs/unified_config.py) *

### Planner Response

现在我已经对项目架构有了足够的了解，可以更新 GEMINI.md 文件了。让我先查看一下 peft_hyper 目录的内容，了解 Hyper-LoRA 系统的结构。

*Edited relevant file*

### Planner Response

GEMINI.md 已成功更新完毕。让我用 notify_user 告知用户。

### User Input

这个项目音频和视频画面是如何融合的，是step by step内容合成还是，融合音视频的信息做内容理解再给用户输出的

### Planner Response

这是个很好的研究问题！让我深入阅读核心代码来给你准确的答案。

*Viewed [unified_arch.py](file:///Users/youcheng/PycharmProjects/Crab/models/unified_arch.py) *

### User Input

Continue

### Planner Response

基于刚才已读取的代码，我可以给你一个准确的分析。

---

## 🎯 Crab 的音视频融合方式：**并行编码 + LLM 统一理解**

这个项目采用的是**"独立编码，Token 拼接，LLM 统一理解"**的融合范式，不是 step-by-step 内容合成，而是**让 LLM 同时"看到"音频和视频信息，再做联合推理输出**。

### 核心流程（来自 [prepare_multimodal_inputs()](cci:1://file:///Users/youcheng/PycharmProjects/Crab/models/unified_arch.py:216:4-405:24)）

```
步骤 1：独立编码（并行）
  视频帧  → VisualEncoder(CLIP) → VLProjector(Q-Former+MLP) → 32个视觉Token
  音频流  → AudioEncoder(BEATs) → ALProjector(Q-Former+MLP) → 32个音频Token

步骤 2：Token序列拼接（融合点）
  [系统提示] + [音频Token×32] + [视频Token×32] + [用户指令] → 一条序列

步骤 3：LLM统一推理（自注意力融合）
  LLaMA-2 对整条序列做 Self-Attention → 文本输出

步骤 4（仅AVS任务）：分割后处理
  LLM输出的 <mask> Token → SegModule → 分割Mask
```

### 关键代码证据

在 [prepare_multimodal_inputs()](cci:1://file:///Users/youcheng/PycharmProjects/Crab/models/unified_arch.py:216:4-405:24) 中（L276-L321），代码用 `inputs_embeds_seg` 列表逐段拼接：

```python
# 遇到 <audio> 占位符时，替换为音频embedding
if special_token == '<audio>':
    feature = self.encode_audio(batch_X_modals[i][special_token], ...)
    inputs_embeds_seg.append(feature)

# 遇到 <video> 占位符时，替换为视觉embedding
elif special_token == '<video>':
    vit_feature_list, qformer_feature_list = self.encode_video(...)
    feature = qformer_feature_list[-1]  # 取最后一层Q-Former特征
    inputs_embeds_seg.append(feature)

# 最终拼接成完整序列
inputs_embeds_seg = torch.cat(inputs_embeds_seg, dim=0)
```

最终输入 LLM 的 embedding 序列长度 = 文本Token数 + 32（音频）+ 32（视频）。

---

### 融合发生在哪里？

| 阶段 | 融合方式 | 说明 |
|------|---------|------|
| 编码阶段 | **无融合** | 音频和视频完全独立通过各自的 Encoder + Q-Former |
| 投影阶段 | **无融合** | 各自用 VLProjector / ALProjector 压缩到 LLM 维度 |
| **LLM阶段** | **✅ 隐式融合（核心）** | 所有Token在 Transformer Self-Attention 中统一交互，跨模态关联在这里产生 |
| 分割阶段 | AVS任务专用 | LLM 输出的 `<mask>` Token 携带了已融合的音视频语义，再驱动 SegModule |

---

### 其最创新之处：**Hyper-LoRA 任务感知**

音视频的"显式合作（Explicit Cooperation）"体现在：不同任务（AVE、AVVP、AVS…）会激活不同的 Hyper-LoRA 权重，从而让 LLM 的注意力模式针对该任务特化，**隐式控制音视频 Token 的交互方式**。

---

### 简单结论

> **不是 step-by-step 合成**，而是 **"晚融合（Late Fusion）via LLM Self-Attention"**——音频和视频各自压缩成 32 个 Token 后直接拼成一条序列，交给 LLM 做端到端的跨模态联合推理，融合发生在 Transformer 的每一层注意力计算中。

### User Input

参考项目1:模型的架构思路https://github.com/Ola-Omni/Ola
参考项目2:数据处理方式https://github.com/mira-ai-lab/MUSIC-AVQA-R
@[/Users/youcheng/PycharmProjects/Crab/20231513079-胡友成-融合音频信息的视觉内容理解模型研究.doc]结合我的思路，可以不可以改造这个项目作为我的学位论文

*Read URL content from /Ola-Omni/Ola *

*Read URL content from /mira-ai-lab/MUSIC-AVQA-R *

*User accepted the command `cd /Users/youcheng/PycharmProjects/Crab && python3 -c "
import subprocess
result = subprocess.run(['textutil', '-convert', 'txt', '-stdout', '20231513079-胡友成-融合音频信息的视觉内容理解模型研究.doc'], capture_output=True, text=True)
print(result.stdout[:8000])
"`*

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

*Viewed content chunk*

*Read URL content from /abs/2501.05067 *

### User Input

Continue

### Planner Response

好，现在我已经读完了所有资料，来给你一个完整深入的分析。

---

## 📋 可行性分析：以 Crab 为基础改造学位论文

### 一、你的论文方案核心（读完中期报告）

你的研究框架是：
> **基座模型（Ola-7B）** + **去偏机制（MCCD）** + **数据重写（MUSIC-AVQA-R 思路）** → 提升音频在 AVQA 任务中的贡献度

---

### 二、Crab vs. Ola — 谁更适合作为你的基座？

| 维度 | **Crab（GeWu-Lab自研）** | **Ola（外部）** |
|------|------|------|
| 与课题相关性 | ✅ 天然支持 MUSIC-AVQA 任务 | ⚠️ 需要重新适配 AVQA 输出头 |
| 基础 LLM | LLaMA-2-7B（相对旧） | Qwen2.5（更现代，推理更强）|
| 音视频融合 | Late Fusion via LLM | Progressive Alignment（更明确的跨模态建模）|
| 代码可操控性 | ✅ 你在本实验室，可深度修改 | ⚠️ 外部代码，改造难度更高 |
| 论文答辩立场 | ⚠️ "改造自己导师的工作"，创新点说明需更充分 | ✅ 引入外部优秀工作加以改进，站位更自然 |
| Hyper-LoRA | ✅ 已内置多任务自适应微调 | ❌ 无 |
| 分割任务支持 | ✅ 有，适合扩展 | ❌ 无 |

---

### 三、**强烈建议的方案：以 Crab 为基础，做针对性改造**

这不是两个方案选一个，而是你其实**可以将 Ola 的架构思路"移植"到 Crab**，同时保留 Crab 现有的工程优势。改造路径如下：

---

#### 🔧 改造方向 1：数据层（来自 MUSIC-AVQA-R）

Crab 的 [dataset/AVQA.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/dataset/AVQA.py:0:0-0:0) 和 [dataset/unified_dataset.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/dataset/unified_dataset.py:0:0-0:0) 已经有 MUSIC-AVQA 的数据加载逻辑。你需要做的是：

1. **问题重写**：用 LLM（GPT-4o/Qwen）对 MUSIC-AVQA 测试集问题做多样化改写，构建 MUSIC-AVQA-R 风格的鲁棒测试集
2. **Head/Tail 划分**：按答案分布的 `1.2×μ(a)` 阈值划分常见/罕见问题子集，用于评估泛化性
3. 这部分工作量明确、创新点清晰，且**不依赖模型改造**，可以先启动

---

#### 🔧 改造方向 2：模型层（来自 Ola 的架构思路）

Ola 最关键的创新是**渐进式对齐训练（Progressive Alignment）**：Language → Image → Video → Audio，视频作为视觉和音频的"桥梁"。

对应到 Crab，可以：

- **强化音视频时序对齐**：Crab 目前的 [VLProjector](cci:2://file:///Users/youcheng/PycharmProjects/Crab/models/multimodal_encoder.py:86:0-143:28) 和 [ALProjector](cci:2://file:///Users/youcheng/PycharmProjects/Crab/models/multimodal_encoder.py:188:0-261:27) 是完全独立的，没有显式的跨模态时序对齐。可以在 Q-Former 阶段引入 **Cross-Attention** 让音频 Query 与视频特征交互
- **引入 MCCD 去偏模块**：在 AVQA 的预测头之前，加入 MCCD（多方面循环协同去偏）分支，分别建立视觉/音频/语言单模态预测分布，与多模态融合结果对比优化。MCCD 是 **plug-and-play** 的，可以较干净地嫁接

---

#### 🔧 改造方向 3：评估体系

在 Crab 现有的 [scripts/finetune/inference_hyper_lora.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/scripts/finetune/inference_hyper_lora.py:0:0-0:0) 推理框架上，增加：
- Head/Tail 子集分开评分
- 单模态消融（去掉音频输入，看精度下降幅度 → 证明音频贡献度提升）

---

### 四、改造后【你的论文贡献点】是什么？

这样改造之后，你的论文三个贡献点非常清晰：

> **C1（数据）**：构建 MUSIC-AVQA-R 鲁棒评估集，引入 Head/Tail 划分方法，揭示现有模型的偏置问题
>
> **C2（模型）**：在 Crab 统一音视频框架中引入显式音视频 Cross-Attention 对齐模块，增强音频信息贡献
>
> **C3（训练）**：将 MCCD 去偏策略嫁接到 Crab 的 AVQA 训练目标中，通过对比损失缓解模态偏置

---

### 五、风险与建议

> [!WARNING]
> **与导师确认**：Crab 是导师组的工作，在其上做论文改造，需明确和导师确认这种"在自研系统上扩展"的定位是否被认可为独立研究贡献。

> [!TIP]
> **建议先做 C1（数据）**：数据构建立竿见影、无模型改动风险，可以最快产出实验结果，作为 3月内的短期目标。

> [!NOTE]
> Crab 本身的 Hyper-LoRA 是你的隐性优势——Ola 没有多任务自适应机制，你在 Crab 上改造后，可以同时在多个任务上验证鲁棒性，这比 Ola 方案在任务覆盖面上更宽。

---

### 总结

**结论：可以，而且 Crab 比 Ola 更适合作为你论文的改造基础。** 理由是：代码完全可控、MUSIC-AVQA 任务已经内置支持、Hyper-LoRA 是额外优势。Ola 的架构思路（音视频桥梁对齐、渐进式训练）和 MUSIC-AVQA-R 的去偏方法（MCCD + 数据重写）可以直接"思路移植"到 Crab 中改造，而不必切换基座模型。

### User Input

生成一个可行性的方案给我

### User Input

Continue

*Edited relevant file*

### User Input

Continue

### Planner Response

方案已生成完毕！请查看：

### User Input



### Planner Response

用户已批准方案，无后续请求，静候下一步指令即可。如果你准备好开始动手实施，可以告诉我从哪个创新点开始（推荐先从 **C1 数据改造**启动，风险最低、最快出结果），我可以直接帮你写代码。

### User Input

我现在需要怎么一步一步实现这个计划

### Planner Response

好，按照已批准的方案，我给你一个**按先后顺序、可以立即动手**的执行路线图。

---

## 🗺️ 执行路线图（共 4 个阶段）

---

### **阶段 0：环境准备与基线确认**（第 1 周，先做）

这是最关键的起点，确认 Crab 在 MUSIC-AVQA 上跑通，才有基线可以对比。

**Step 1** — 确认数据就位
```bash
# 检查 dataset/AVQA.py 中 data_root 路径是否正确配置
# MUSIC-AVQA 原始数据需要从官方下载：
# https://github.com/GeWu-Lab/MUSIC-AVQA
```

**Step 2** — 跑一次 AVQA 基线推理
```bash
# 修改 inference_hyper_lora.sh，开启 avqa_task=True
bash scripts/finetune/inference_hyper_lora.sh
```

**Step 3** — 记录基线分数（按问题类型分：EXIST / LOC / CNT / COMP / TEMP）

---

### **阶段 1：C1 数据改造**（第 2-4 周）

目标：构建 MUSIC-AVQA-R 风格的鲁棒数据集。

**Step 1** — 下载 MUSIC-AVQA-R 原始数据
```bash
# 从 MUSIC-AVQA-R 仓库获取重写后的问题 JSON
# https://github.com/mira-ai-lab/MUSIC-AVQA-R
```

**Step 2** — 新建数据加载文件 `dataset/avqa_robust.py`
- 参考 [dataset/AVQA.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/dataset/AVQA.py:0:0-0:0) 的结构
- 加入 `head` / `tail` 标签字段
- 支持加载重写后的问题

**Step 3** — 修改推理评估脚本
- 在 [scripts/finetune/inference_hyper_lora.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/scripts/finetune/inference_hyper_lora.py:0:0-0:0) 末尾增加按 `head` / `tail` 分组计算 Accuracy 的逻辑

**Step 4** — 用 Crab 原版在 MUSIC-AVQA-R 上评估
- 对比 head 和 tail 精度差异 → **这就是你论文第 3 章"偏置分析"的实验数据**

---

### **阶段 2：C2 模型改造**（第 5-9 周）

在 [models/multimodal_encoder.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/models/multimodal_encoder.py:0:0-0:0) 中新增 `AVCrossAttentionFusion`，并集成到前向传播。

**Step 1** — 实现模块（约 30 行代码）

```python
# 在 models/multimodal_encoder.py 末尾新增
class AVCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=4096, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, audio_feat, video_feat):
        fused, _ = self.cross_attn(audio_feat, video_feat, video_feat)
        return self.norm(audio_feat + fused)
```

**Step 2** — 在 `UnifiedMetaModel.init_multimodal_modules()` 中注册
```python
# 在 audio_branch 初始化块末尾加入
self.av_crossattn = AVCrossAttentionFusion(d_model=d_model)
```

**Step 3** — 在 `UnifiedMetaForCausalLM.encode_audio()` 中调用
```python
# encode_audio 返回前，融合视频特征
audio_feature = self.get_model().av_crossattn(audio_feature, video_feature)
```

**Step 4** — 重新微调 AVQA 任务，对比 C2 加入前后的 head/tail 精度

---

### **阶段 3：C3 训练改造**（第 10-12 周）

在 [models/unified_llama.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/models/unified_llama.py:0:0-0:0) 的 [forward()](cci:1://file:///Users/youcheng/PycharmProjects/Crab/models/multimodal_encoder.py:74:4-83:31) 中加入 MCCD 去偏损失。

**Step 1** — [configs/unified_config.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/configs/unified_config.py:0:0-0:0) 新增参数
```python
use_mccd: bool = field(default=False)
mccd_lambda: float = field(default=0.1)
```

**Step 2** — 实现单模态遮蔽前向
- 音频遮蔽：将 `<audio>` Token 位置的 embedding 替换为零向量
- 视频遮蔽：将 `<video>` Token 位置的 embedding 替换为零向量

**Step 3** — 加入 KL 散度去偏损失
```python
if self.config.use_mccd:
    kl = F.kl_div(F.log_softmax(av_logits,-1), F.softmax(v_logits,-1)) \
       + F.kl_div(F.log_softmax(av_logits,-1), F.softmax(a_logits,-1))
    loss = ce_loss + self.config.mccd_lambda * kl
```

**Step 4** — 重新跑 AVQA 专项微调 + 评估，形成最终对比表

---

## 📋 你现在立刻要做的第一件事

> **「运行 Crab 的 AVQA 基线评估，记录分数」**

这是整个论文的锚点，**所有后续改进都需要和它对比**。你是否已经有 Crab 的微调权重？如果有，我可以直接帮你写评估脚本中的 Head/Tail 分组统计代码。

### User Input

不计成本，如何一周内实现

### Planner Response

**一周极限冲刺计划**，核心原则：**能用 API 的不手写，能复用权重的不重新训练，只做最小可验证的改动**。

---

## ⚡ 7 天冲刺计划

### Day 1（周一）：环境 + 基线
- 确认 Crab finetune 权重已有（HuggingFace 可直接下载）
- 跑通 MUSIC-AVQA 推理，记录原始 Accuracy
- 安装 `openai` / `dashscope` 包，准备问题重写 API

### Day 2（周二）：C1 数据 — 用 GPT-4o 批量重写
不手写改写逻辑，直接调 API：

```python
import openai, json

client = openai.OpenAI(api_key="YOUR_KEY")
questions = json.load(open("music_avqa_test.json"))

def rewrite(q):
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user",
                   "content":f"Rephrase this question differently but keep the same meaning: '{q}' Output only the rephrased question."}]
    )
    return r.choices[0].message.content

# 批量重写（~9000条，GPT-4o约花费$3-10）
for item in questions:
    item["question_r"] = rewrite(item["question"])

json.dump(questions, open("music_avqa_r.json","w"))
```

然后按答案频率做 Head/Tail 划分（约 20 行代码）。**当天可完成 C1 全部工作。**

### Day 3（周三）：C2 模型改造（最小化实现）

跳过渐进式训练，直接在现有 finetune 权重上**插入 AV-CrossAttn 做 LoRA 微调**：

```python
# models/multimodal_encoder.py 末尾加入（约30行）
class AVCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=4096, nhead=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, audio_feat, video_feat):
        fused, _ = self.cross_attn(audio_feat, video_feat, video_feat)
        return self.norm(audio_feat + fused)
```

挂上模块，冻结其他权重，只 finetune 这个新模块 + LoRA。

### Day 4（周四）：C3 MCCD 损失（最小化实现）

在 [unified_llama.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/models/unified_llama.py:0:0-0:0) 加入 30 行损失计算，开启训练。

**同时**：把 Day 2 的数据集 + Day 3 的评估脚本全部跑起来，多个实验并行在多卡上跑。

### Day 5（周五）：全部实验并行运行中，整理结果

| 实验 | GPU | 预计时长 |
|------|-----|--------|
| Crab 基线在 AVQA-R 上评估 | 1 卡 | 2h |
| Crab + AV-CrossAttn 微调 | 2 卡 | 4-6h |
| Crab + MCCD 微调 | 2 卡 | 4-6h |
| Crab + C2 + C3 微调 | 2 卡 | 6-8h |

有钱可直接租 AutoDL / A800，4 个实验同时跑，一天出结果。

### Day 6-7（周末）：整理对比表

| 模型 | Overall | Head | Tail |
|------|---------|------|------|
| Crab 原版 | - | - | - |
| + AV-CrossAttn | - | - | - |
| + MCCD | - | - | - |
| + 全部（Crab-AV）| - | - | - |

---

## 💡 节省时间的关键决策

| 原计划 | 一周极限版 |
|--------|-----------|
| 手动重写问题（3周） | GPT-4o API 批量处理（1天）|
| 渐进式多阶段预训练 | 直接在现有权重上插入新模块微调 |
| 全参数微调 | **LoRA 微调**（训练时间减少 60%）|
| 逐步调参 | 先用默认超参跑出结果，有差距再调 |

---

## 🚦 你现在第一步

**你现在有 Crab 的 finetune 权重吗？**（HuggingFace 上的 `finetune_weights.bin`）

如果有，我立刻帮你写 Day 2 的 GPT-4o 批量重写脚本和 Head/Tail 划分代码，今天就能启动。

### User Input

请结合ola的工作，如果不结合好的话，请给我不结合的理由

### Planner Response

这是个很关键的决策点，我给你**直接的判断**：

---

## 建议：**不要**直接把 Ola 代码库引入，但要**引用 Ola 的设计思想**

---

### ❌ 不直接使用 Ola 代码库的理由

**理由 1：换基座 = 换工程基础，一周内不可能完成**

Ola 的代码架构与 Crab 完全不同：
- Ola 基于 Qwen2.5 的 LM + 自定义 streaming 解码器
- MUSIC-AVQA 任务在 Ola 里根本没有接口，要从头写数据加载、评估逻辑
- 你要重新学习一套新的 codebase，光熟悉就要 2-3 天

你在 GeWu-Lab，Crab 里的 [dataset/AVQA.py](cci:7://file:///Users/youcheng/PycharmProjects/Crab/dataset/AVQA.py:0:0-0:0) 已经帮你把 MUSIC-AVQA 接好了。这是不可忽视的工程优势。

**理由 2：Ola 的 LLM 更强是双刃剑**

Qwen2.5 比 LLaMA-2 强，但这意味着：
- Ola 的基线分数更高，你的"改进幅度"可能更难体现
- Crab 基线较低，MCCD + AV-CrossAttn 的提升反而更显著、更容易写论文

**理由 3：你的论文贡献点不依赖 Ola 的架构**

你的三个创新点（去偏数据集、音视频对齐、MCCD 损失）本质上是**方法层面的贡献**，和底层 LLM 是 LLaMA 还是 Qwen 关系不大。在 Crab 上验证同样有效。

---

### ✅ 要"结合 Ola"的方式：**思想引用，不是代码迁移**

Ola 给你的价值是**两个核心 idea**，已经体现在你的方案里了：

| Ola 的思想 | 在你方案中的体现 |
|-----------|----------------|
| **视频作为音视频桥梁** | C2 中的 `AVCrossAttentionFusion`：让音频 Query 通过 Cross-Attention 感知视频特征 |
| **渐进式对齐训练** | C2 的训练策略：先对齐各模态，再做 AVQA 专项微调 |

论文里这样写就够了：
> "受 Ola [ref] 渐进式音视频对齐思路的启发，本文在 Crab 的 Q-Former 之后引入音视频交叉注意力模块……"

---

### 最终结论

| | 直接用 Ola 代码库 | 思想移植到 Crab |
|--|--|--|
| 1周内可完成 | ❌ | ✅ |
| MUSIC-AVQA 原生支持 | ❌ 需要重写 | ✅ 已有 |
| 论文中能引用 Ola | ✅ | ✅ 同样可以 |
| 创新点清晰度 | ⚠️ 容易变成"复现 Ola" | ✅ Crab 上的明确扩展 |

**直接用 Ola 代码库会变成复现+微调别人的工作，在 Crab 上做改造才是你自己的贡献。** 已批准的方案（thesis_plan.md）就是最优路径，不需要改。

### User Input

以今天时间为开始，我第一天需要做什么，以及我需要什么样的软硬件配置

### Planner Response

现在是 **2月25日（周二）凌晨**，下面是你今天（Day 1）需要做的所有事情。

---

## 🖥️ 软硬件配置需求

### 硬件（最低 / 推荐）

| 资源 | 最低（能跑推理） | 推荐（能跑训练） |
|------|----------------|----------------|
| GPU | 1× A100 40GB | 2× A100 80GB |
| 内存 | 64 GB RAM | 128 GB RAM |
| 存储 | 200 GB SSD | 500 GB SSD |
| 网络 | 能访问 HuggingFace | 同左 |

> 如果实验室没有这个配置，立刻去 **[AutoDL](https://www.autodl.com)** 租 —— A100 80GB 约 ¥5-7/小时，一天推理实验约 ¥30-50。

---

### 软件环境

```bash
# 基础环境（Crab 官方要求）
Python == 3.9
PyTorch == 2.1.0
CUDA == 11.8 或 12.1
transformers == 4.37.2
deepspeed == 0.12.6

# 额外需要安装（Day 2 问题重写用）
pip install openai
pip install huggingface_hub
```

---

## 📋 今天（Day 1）的具体任务清单

### ✅ 任务 1：下载 Crab 权重（约 1-2 小时）

```bash
# 安装 HuggingFace CLI
pip install huggingface_hub

# 下载 finetune 权重（论文改造的起点）
huggingface-cli download ahsgdxhs/Crab \
  finetune_weights.bin \
  --local-dir ./ckpt/

# 同时下载基础模型权重
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir ./pretrain/llama2/

huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir ./pretrain/clip/
```

### ✅ 任务 2：下载 MUSIC-AVQA 数据集（约 1 小时）

```bash
# 从官方仓库获取标注文件
git clone https://github.com/GeWu-Lab/MUSIC-AVQA
# 视频/音频文件从 Google Drive 下载（见仓库 README）
```

### ✅ 任务 3：配置推理脚本（约 30 分钟）

修改 [scripts/finetune/inference_hyper_lora.sh](cci:7://file:///Users/youcheng/PycharmProjects/Crab/scripts/finetune/inference_hyper_lora.sh:0:0-0:0)：
```bash
# 设置以下参数
avqa_task=True
ckpt_dir="./ckpt"                    # 刚才下载的权重路径
llm_path="./pretrain/llama2"
vit_ckpt_path="./pretrain/clip"
```

### ✅ 任务 4：跑基线推理，记录分数（约 2-4 小时）

```bash
bash scripts/finetune/inference_hyper_lora.sh
```

推理结束后，把以下数字记下来（这是你论文 Table 1 的第一行）：

| 指标 | 数值 |
|------|------|
| Overall Accuracy | ? |
| EXIST | ? |
| LOC | ? |
| CNT | ? |
| COMP | ? |
| TEMP | ? |

### ✅ 任务 5（今晚）：申请 OpenAI API Key

去 [platform.openai.com](https://platform.openai.com) 充值 $10，明天 Day 2 批量重写问题要用。

---

## ⏰ 今天的时间分配

| 时间 | 任务 |
|------|------|
| 现在 - 早上 | 挂起权重下载 + 数据下载（后台跑） |
| 上午 | 配置环境，安装依赖包 |
| 下午 | 跑基线推理 |
| 晚上 | 记录基线分数，申请 OpenAI API |

---

**你现在最需要确认的一件事：实验室的服务器上有没有 A100？** 如果没有，先去 AutoDL 开一台机器，其他都可以并行进行。

### User Input

我准备使用阿里云，autodl的SSD大小不够

### Planner Response

好选择！阿里云存储更灵活。以下是针对阿里云的配置建议：

---

## 阿里云配置推荐

### 实例选择

| 场景 | 推荐实例 | GPU | 价格参考 |
|------|---------|-----|---------|
| 推理（基线评估） | `ecs.gn7i-c16g1.4xlarge` | 1× A10 24GB | ≈ ¥8/小时 |
| 训练（微调） | `ecs.gn7i-c32g1.8xlarge` | 2× A10 24GB | ≈ ¥16/小时 |
| 高配训练 | `ecs.gn8i-c32g1.4xlarge` | 2× A100 40GB | ≈ ¥40/小时 |

> 推荐直接上 **2× A100 40GB**，训练时间减半，总费用其实差不多。

---

### 存储配置（关键）

```
系统盘：  100 GB（装系统和代码）
数据盘：  挂载 1个 500 GB 高效云盘（约 ¥0.35/GB/月）
          └─ 用于存放：
             ├── 模型权重（LLaMA-2: ~14GB, CLIP: ~2GB, Crab: ~1GB）
             ├── MUSIC-AVQA 数据（视频+音频约 100-150 GB）
             └── 实验输出
```

### 挂载数据盘命令（买完实例后执行）

```bash
# 查看数据盘设备名
lsblk

# 格式化并挂载（假设设备为 /dev/vdb）
mkfs.ext4 /dev/vdb
mkdir /data
mount /dev/vdb /data

# 开机自动挂载
echo "/dev/vdb /data ext4 defaults 0 0" >> /etc/fstab
```

---

### 镜像选择

创建实例时选：
- **镜像**：`PyTorch 2.1.0 + CUDA 11.8`（阿里云预装镜像，省去环境配置时间）
- **地域**：选离你最近的，华东/华北延迟最低

---

### 今天的行动顺序

1. **现在**：去阿里云控制台创建实例 + 挂载 500GB 数据盘
2. **实例就绪后**：把 `/data` 目录用于存放所有数据和权重
3. **运行环境安装**：
```bash
pip install -r /path/to/Crab/requirements.txt
pip install openai huggingface_hub
```

4. **从 HuggingFace 下载权重**（阿里云出网速度快，约 30 分钟）：
```bash
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像加速
huggingface-cli download ahsgdxhs/Crab --local-dir /data/ckpt/
```

---

实例准备好后告诉我，我帮你直接写好一键配置脚本。

### User Input

如何再时间线内，节约服务器租赁的费用呢

### Planner Response

核心原则：**GPU 只在真正计算时开着，其他时间全部关机。**

---

## 💰 省钱策略

### 策略 1：本地 Mac 写代码，云端只跑计算

你的 MacBook 完全够用于：
- 写代码、调试逻辑
- 处理 JSON 数据（问题重写、Head/Tail 划分）
- 分析实验结果

**只有以下操作才需要开云服务器：**
- 模型推理（评估基线）→ **约 2-4 小时**
- LoRA 微调训练 → **约 4-8 小时/次**

---

### 策略 2：用抢占式实例（便宜 60-70%）

阿里云"抢占式实例"价格约为按需实例的 **1/3**：

| 实例 | 按需价格 | 抢占式价格 |
|------|---------|----------|
| 2× A10 24GB | ¥16/h | ¥5/h |
| 2× A100 40GB | ¥40/h | ¥12/h |

> ⚠️ 抢占式实例可能被回收，但训练中途保存 checkpoint 即可恢复。

---

### 策略 3：数据存到 OSS，实例随时释放

```bash
# 每次训练结束后，把结果传到 OSS，然后释放实例
ossutil cp -r /data/results/ oss://你的bucket/results/
ossutil cp /data/ckpt/best.bin oss://你的bucket/ckpt/

# 下次开实例时，从 OSS 拉回来
ossutil cp -r oss://你的bucket/ /data/
```

OSS 存储费用：500GB ≈ **¥0.12/GB/月 ≈ ¥60/月**，远比挂着实例省钱。

---

### 策略 4：批量排队，一次开机跑完所有实验

**不要**每次实验单独开机。写好多个脚本，**一次开机顺序执行**：

```bash
# 一次性跑完所有实验
bash run_all.sh &
# run_all.sh 内容：
# 1. 基线推理
# 2. AV-CrossAttn 微调
# 3. MCCD 微调
# 4. 完整方案微调
# 5. 所有推理评估
# 6. 上传结果到 OSS
# 7. 自动关机
sudo shutdown -h now  # 实验结束自动关机
```

---

### 费用估算（7天总花销）

| 操作 | 时长 | 预计费用（抢占式 A100） |
|------|------|----------------------|
| Day 1：基线推理 | 3h | ¥36 |
| Day 2：数据处理（Mac本地） | - | ¥0 |
| Day 4：C2 微调 | 8h | ¥96 |
| Day 5：C3 微调 | 8h | ¥96 |
| Day 6：全部评估 | 4h | ¥48 |
| OSS 存储（1个月） | - | ¥60 |
| **总计** | | **≈ ¥336** |

**7天全程约 ¥300-400，不到 ¥500。**

---

### 今天要做的准备（Mac 本地，免费）

数据处理、问题重写脚本全部在 Mac 上写好、测试通过，等服务器开机后直接跑，不浪费 GPU 时间：

```bash
# 本地先安装 openai，测试脚本逻辑
pip install openai
python test_rewrite.py  # 用5条样本测试，确认没问题
```

### User Input

但是我服务器下载模型权重还有数据不也是需要时间的吗

### Planner Response

非常对！这是个关键问题。解决方案是：**下载和计算分开，用便宜的 CPU 实例下载，GPU 只在计算时开。**

---

## 解决方案：二阶段策略

### 阶段 A：用 CPU 实例下载数据（便宜）

阿里云 **ecs.c7.xlarge**（4核 8GB 内存，**无 GPU**）：
- 价格：约 **¥0.4/小时**（是 A100 的 1/30）
- 用途：专门用来下载权重和数据集，挂着跑一晚上都没关系

```bash
# 开一台 CPU 实例，挂载好 OSS 或大数据盘
# 跑以下下载命令（可以整晚挂着）
export HF_ENDPOINT=https://hf-mirror.com

# 下载 LLaMA-2（约 14GB）
huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir /data/pretrain/llama2/

# 下载 CLIP（约 2GB）  
huggingface-cli download openai/clip-vit-large-patch14 \
  --local-dir /data/pretrain/clip/

# 下载 BEATs 音频编码器（约 300MB）
# 从 Crab README 的链接下载

# 下载 Crab finetune 权重（约 5-10GB）
huggingface-cli download ahsgdxhs/Crab \
  --local-dir /data/ckpt/
```

挂一晚上（10小时），费用 **¥4**，全部下完。

---

### 阶段 B：数据下完后，快照成镜像

下载完成后，**把数据盘制作成快照**（阿里云控制台一键操作）：

```
数据盘快照 → 下次创建 GPU 实例时直接挂载这个快照
```

这样每次开新 GPU 实例，数据秒到，不需要重新下载。

---

### 整体流程

```
今晚  → 开 CPU 实例（¥0.4/h）→ 挂着下载数据 → 睡觉
明早  → 下载完成，制作快照 → 关掉 CPU 实例
白天  → 代码写好后，开 GPU 实例 → 挂载快照 → 直接训练
训练完 → 结果传 OSS → 关 GPU 实例
```

---

### 费用对比

| 方案 | 下载耗时 | 费用 |
|------|---------|------|
| ❌ 用 A100 下载 | 2h | ¥24-80 |
| ✅ 用 CPU 实例下载 | 8h（可挂夜） | ¥3 |

**今晚就可以开 CPU 实例挂下载，明天白天数据就绪，代码也写好了，直接开 GPU 跑。**

### User Input

服务器的网络带宽如何选择，服务器实例的地域要怎么选

### Planner Response

---

## 地域选择

**推荐：华东2（上海）**

| 地域 | 推荐理由 |
|------|---------|
| **华东2 上海** ✅ | hf-mirror.com 镜像站速度最快；GPU 实例资源充足；OSS 同地域传输免费 |
| 华北2 北京 | 次选，GPU 抢占式实例较多 |
| ❌ 海外地域 | HuggingFace 直连快，但 OSS/VPC 内网互通复杂，不推荐 |

> **关键原则：CPU 实例、GPU 实例、OSS Bucket 三者必须在同一地域**，这样实例间数据传输走内网，**速度极快且免费**。

---

## 带宽选择

### CPU 下载实例（用来拉模型权重）
- **计费方式：按使用流量计费**（不要选按固定带宽）
- **峰值带宽：选最大，200Mbps 或更高**
- 原因：按流量计费时你只为实际下载的数据付钱，带宽拉满只影响下载速度，不影响费用

```
下载 LLaMA-2 (14GB)：
- 100Mbps ≈ 10 分钟
- 10Mbps  ≈ 1.7 小时
拉满带宽意味着整晚能下完几百GB
```

### GPU 训练实例（用来训练推理）
- **计费方式：按固定带宽**
- **带宽：1-5 Mbps 就够**
- 原因：训练数据从本地快照盘读取，不走公网；只有 SSH 连接和上传结果到 OSS 需要外网，流量极小

---

## 总结配置单

| 实例 | 地域 | 带宽选择 |
|------|------|---------|
| CPU 下载实例 | 华东2 上海 | 按流量计费 + 200Mbps 峰值 |
| GPU 训练实例 | 华东2 上海 | 按固定带宽 + 5Mbps |
| OSS Bucket | 华东2 上海 | — |

**今晚操作顺序：先建 OSS Bucket（华东2），再建 CPU 实例（同地域），挂载数据盘，开始下载。**

### User Input

华东的实例能快速下载huggingface上的模型还有数据吗？安装环境依赖包会慢吗

### Planner Response

**直接回答：HuggingFace 不能直连，但有镜像站完美解决；pip 安装用国内源也很快。**

---

## HuggingFace 下载

国内阿里云实例**无法直接访问 HuggingFace**（被墙），但有官方镜像：

```bash
# 设置 HuggingFace 国内镜像（hf-mirror.com，国内最稳定的镜像）
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常下载，速度可达 50-100MB/s
huggingface-cli download ahsgdxhs/Crab --local-dir /data/ckpt/
```

> `hf-mirror.com` 是国内社区维护的镜像，同步及时，速度稳定，阿里云华东实例访问速度约 **50-100 MB/s**（200Mbps 带宽时）。

---

## pip 安装依赖包

**用阿里云镜像源，速度非常快（10-50 MB/s）**：

```bash
# 永久设置 pip 镜像为阿里云源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com

# 然后直接安装 Crab 依赖
pip install -r requirements.txt
```

全部依赖安装完约 **5-10 分钟**，不会慢。

---

## 今晚操作的完整命令（开机后一次性复制粘贴）

```bash
# 1. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 2. 安装 huggingface 工具
pip install huggingface_hub -q

# 3. 安装 Crab 依赖（后台跑，不阻塞）
pip install -r /root/Crab/requirements.txt &

# 4. 同时开始下载权重（并行进行）
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir /data/pretrain/llama2/ &
huggingface-cli download openai/clip-vit-large-patch14 --local-dir /data/pretrain/clip/ &
huggingface-cli download ahsgdxhs/Crab --local-dir /data/ckpt/ &

# 5. 等所有任务完成
wait
echo "全部下载完成！"
```

把这段脚本保存成 `setup.sh`，开机后一条命令搞定，挂着睡觉。

### User Input

如果用非大陆的实例会不会省事一点呢

### Planner Response

省事，但有一个关键代价，我来客观分析：

---

## 非大陆实例 vs 大陆实例

| 维度 | **非大陆（香港/新加坡）** | **大陆（华东上海）** |
|------|--------------------------|---------------------|
| HuggingFace 下载 | ✅ 直连，无需镜像 | ❌ 需要 hf-mirror.com |
| pip 安装 | ✅ PyPI 官方源可用 | 需要换阿里云镜像 |
| GPU 实例可用性 | ⚠️ **资源少，A100 经常没货** | ✅ 资源充足 |
| 价格（GPU） | ❌ 贵 20-40% | ✅ 便宜 |
| 训练速度 | 相同 | 相同 |
| 与 OSS 传输 | ⚠️ 跨地域有费用 | ✅ 同地域免费 |

---

## 直接结论

**不推荐用非大陆实例**，原因是：

1. **GPU 资源紧缺**：香港/新加坡的 A100 实例几乎常年缺货，你要跑训练抢不到机器
2. **价格更贵，GPU 才是大头成本**：下载省的那点时间配置成本，被 GPU 溢价完全抵消
3. **镜像配置只需 2 条命令**，10 秒钟的事，并不真的"麻烦"

```bash
# 大陆实例，这两行就解决所有国内访问问题
export HF_ENDPOINT=https://hf-mirror.com
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
```

**如果你坚持要用境外实例**，选 **新加坡** 而不是香港（新加坡 HuggingFace 速度更快，实例资源稍多），但 GPU 缺货问题一样存在。

**建议还是华东上海，用镜像，没有任何实质性麻烦。**

### User Input

OSS传输是什么，如何检查是否是跨地域

### Planner Response

## OSS 是什么

**OSS（Object Storage Service）** 是阿里云的对象存储服务，类似网盘，用来持久化存储文件。在我们的使用场景里，它的作用是：

```
GPU 实例（临时）  →  训练完  →  把权重/结果传到 OSS（永久）→  关机
下次开 GPU 实例  →  从 OSS 把文件拉回来  →  继续工作
```

---

## 跨地域的问题

**同地域传输**：ECS 实例 ↔ OSS Bucket 在同一个地域（如都在华东2），走阿里云内网，**免费且极快**

**跨地域传输**：ECS 在华东2，OSS Bucket 在华北2，走公网，**收费且慢**

---

## 如何检查是否跨地域

**方法 1：看控制台**

- 打开 [OSS 控制台](https://oss.console.aliyun.com) → 点击你的 Bucket → 看"地域"字段
- 打开 [ECS 控制台](https://ecs.console.aliyun.com) → 看实例的"地域/可用区"

两者地域名称一致 = 同地域 ✅

**方法 2：用 Endpoint 判断**

访问 OSS 时使用**内网 Endpoint**（包含 `-internal`），阿里云会自动走内网：

```bash
# ✅ 内网访问（同地域免费）
ossutil cp /data/results/ oss://你的bucket/results/ \
  --endpoint oss-cn-shanghai-internal.aliyuncs.com

# ❌ 外网访问（跨地域收费）
ossutil cp /data/results/ oss://你的bucket/results/ \
  --endpoint oss-cn-shanghai.aliyuncs.com
```

---

## 简单记忆

> 创建 OSS Bucket 时，**地域选和 ECS 一模一样的**，之后用带 `-internal` 的 Endpoint 传文件，就永远不会跨地域。

### User Input

如果使用Auto DL平台里的RTX PRO 6000/96G的显卡怎么样

### Planner Response

**RTX PRO 6000 96GB 非常适合这个项目**，逐项分析：

---

## 显卡评估

| 指标 | RTX PRO 6000 96GB | A100 80GB（对比） |
|------|-------------------|-----------------|
| 显存 | ✅ 96GB（更大） | 80GB |
| 架构 | Blackwell（最新一代） | Ampere |
| FP16 算力 | ~180 TFLOPS | 312 TFLOPS |
| 适合场景 | 大 batch、大模型推理 | 训练速度更快 |
| AutoDL 价格 | 查看当前价格 | 约 ¥8-12/h |

对你的任务来说：
- **LLaMA-2-7B + CLIP + BEATs + LoRA 微调** → 约占 40-50GB 显存，96GB 绰绰有余，可以开很大的 batch size 加速训练
- **单卡就够**，不需要多卡分布式，省去 DeepSpeed 配置麻烦

---

## 关于之前说的 AutoDL SSD 不够

你之前说 AutoDL SSD 太小，这是指**系统盘**（默认 50GB）。

解决方案：
```
AutoDL 创建实例时 → 选"数据盘扩容" → 加到 500GB
费用：约 ¥0.04/GB/天，500GB ≈ ¥20/天
```

或者更省钱的做法：
- 系统盘只装环境（50GB 够）
- 挂载 **网络盘（NAS）** 存数据，按量付费

---

## 结论

**RTX PRO 6000 96GB 在 AutoDL 上完全可行**，对比阿里云的优势是：

- 开机更快（AutoDL 镜像更丰富，PyTorch 环境预装好）
- hf-mirror.com 同样可用
- 不需要额外配置 OSS

**存储问题靠扩容数据盘解决就行**，之前不用 AutoDL 的顾虑其实可以消除。

### User Input

阿里云只有这些显卡资源，没有A100

### Planner Response

看了你截图，从阿里云现有选项里：

---

## 可用显卡推荐排名

| 排名 | 显卡 | 显存 | 适合度 | 说明 |
|------|------|------|--------|------|
| 🥇 | **L20加速** | 48GB | ✅ 最推荐 | Ada Lovelace 新架构，48GB 够用，性价比最高 |
| 🥈 | **A10加速** | 24GB | ⚠️ 勉强 | 训练时 batch size 要调小，速度慢一倍 |
| 🥉 | **V100加速** | 32GB | ⚠️ 可用 | 架构老（2017年），速度比 L20 慢约 40% |
| ❌ | T4 / P100 / P4 | 16GB | ❌ | 显存太小，LLaMA-2 + LoRA 放不下 |

---

## 重新比较两个平台

| | **阿里云 L20 48GB** | **AutoDL RTX PRO 6000 96GB** |
|-|---------------------|------------------------------|
| 显存 | 48GB（够用） | 96GB（宽裕） |
| 存储 | 可挂大数据盘，无限制 | 需手动扩容数据盘 |
| 配置复杂度 | 稍复杂（OSS等） | 简单，开箱即用 |
| 价格 | 参考 L20 报价 | 查 AutoDL 当前价格 |

---

## 最终建议

**优先选 AutoDL 的 RTX PRO 6000 96GB**，理由：

1. 96GB 显存比 L20 的 48GB 大一倍，训练更稳
2. AutoDL 开箱即用，PyTorch/CUDA 已预装，省去大量配置时间
3. 存储扩容在 AutoDL 里加数据盘即可解决

阿里云 L20 作为备选，如果 AutoDL 抢不到 RTX PRO 6000 再用。

### User Input

auto DL支持无卡启动，省钱吗

### Planner Response

**非常省钱！这正好解决了之前说的"下载数据时不需要开 GPU"的问题。**

---

## AutoDL 无卡启动的作用

AutoDL 支持**无卡模式启动实例**，此时只有 CPU + 内存 + 存储在运行：

| 模式 | 费用 | 用途 |
|------|------|------|
| **无卡启动** | 仅收存储费（约 ¥0.1-0.3/小时） | 下载数据、写代码、数据处理 |
| **有卡运行** | GPU 全价计费 | 模型训练、推理 |

---

## 完美匹配你的使用场景

```
今晚：无卡启动 → 下载 HuggingFace 模型权重（几小时）→ 费用极低
明天上午：写代码、处理数据依然无卡 → 继续省钱  
需要训练时：控制台手动"开机挂卡" → 开始计费
训练完成：卸载 GPU 或关机 → 停止计费
数据留在实例磁盘上，下次挂卡即可继续
```

---

## 和之前阿里云 CPU 实例方案对比

| | 阿里云 CPU 实例 + OSS | **AutoDL 无卡启动** |
|-|-----------------------|---------------------|
| 下载数据 | CPU 实例 ¥0.4/h | ✅ 更便宜 ¥0.1-0.3/h |
| 数据转移 | 需配置 OSS | ✅ 数据直接在本地磁盘 |
| 配置复杂度 | 高 | ✅ 简单 |
| 存储一致性 | OSS 中转 | ✅ 始终在同一块盘 |

**AutoDL 无卡启动完全替代了之前说的"阿里云 CPU 下载实例 + OSS"方案，更简单、更便宜。**

---

## 建议策略

1. **现在**：AutoDL 开实例，选 RTX PRO 6000，勾选"无卡启动"，扩容数据盘到 500GB
2. **今晚挂着**：无卡模式下载所有权重和数据
3. **需要跑实验时**：控制台点"挂卡开机"，跑完就卸载

### User Input

这个配置如何

### Planner Response

配置基本没问题，但有一处**建议修改**：

---

## 配置评估

| 配置项 | 你的选择 | 评价 |
|--------|---------|------|
| GPU型号 | RTX PRO 6000 96GB | ✅ 完全合适 |
| **GPU数量** | **2卡** | **⚠️ 建议改成 1 卡** |
| 内存 | 220GB | ✅ 充裕 |
| 数据盘 | 500GB | ✅ 够用 |
| 镜像 | PyTorch 2.1.2 / CUDA 11.8 | ✅ 和 Crab 要求（2.1.0）完全兼容 |

---

## 为什么改成 1 卡？

- LLaMA-2-7B + LoRA 微调 **单卡 96GB 完全放得下**，不需要两卡
- 2 卡需要配置 DeepSpeed 多卡分布式，增加调试复杂度
- 2 卡价格 ¥11.96/h，1 卡 ¥5.98/h，**省一半**
- 1 卡训练完全够，如果 batch size 大到撑满再考虑扩

---

## 确认可以创建

改成 1 卡后，其余配置直接创建即可：

```
GPU: RTX PRO 6000 × 1
数据盘: 500GB
镜像: PyTorch 2.1.2 / Ubuntu22.04 / CUDA 11.8  ✅
```