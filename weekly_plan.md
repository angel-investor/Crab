# 论文一周推进计划（2026/03/03 - 03/09）

> **当前进度**：Crab `quick_start` demo 已跑通（含 AVQA、Ref-AVS 调试完成）
> **本周目标**：完成基线评估、C1 数据改造、C2/C3 核心代码实现，并启动论文写作

---

## Day 1（3/3）— 建立评估基线

- [ ] 在完整 MUSIC-AVQA 测试集上跑 Crab 原版推理（`inference_hyper_lora.sh`）
- [ ] 记录原版 Crab 的 AVQA 总体 Accuracy（论文基线数据 `ACC_base`）
- [ ] **音频消融实验**：在 `models/unified_arch.py` 的 `prepare_multimodal_inputs()` 中加 `--mask_audio` 开关，将音频 Token 清零后再次推理
- [ ] 记录 `ACC_no_audio`，计算 `ACC_base - ACC_no_audio`（音频贡献度）
- [ ] 整理两组数字 → 论文第3章"偏置分析"核心素材

---

## Day 2（3/4）— C1 数据改造：问题重写

- [ ] 从 MUSIC-AVQA 测试集 JSON 中提取全部问题（约 9000 条）
- [ ] 编写批量调用 GPT-4o / Qwen2.5 的改写脚本
  - Prompt 原则：保持语义不变，改变句式，打破模板化结构
  - 示例：`"What is the left instrument?"` → `"Among all instruments shown, which one is positioned to the left?"`
- [ ] 抽样人工检查 30 条，确认改写质量
- [ ] 保存为 `music_avqa_test_rewritten.json`

---

## Day 3（3/5）— C1 数据改造：Head/Tail 划分

- [ ] 统计测试集答案频率分布，画直方图（可视化偏置）
- [ ] 按如下公式计算阈值并打标签：
  ```python
  mu_a = mean(answer_counts)
  threshold = 1.2 * mu_a
  head_set = {ans for ans, cnt in answer_counts.items() if cnt > threshold}
  tail_set = {ans for ans, cnt in answer_counts.items() if cnt <= threshold}
  ```
- [ ] 为原始数据集和改写数据集各打上 `head`/`tail` 标签
- [ ] 新建 `dataset/avqa_robust.py` 数据加载器

---

## Day 4（3/6）— 跑偏置分析实验 + 完善评估脚本

- [ ] 修改 `scripts/finetune/inference_hyper_lora.py`，支持 Head/Tail 分组输出 Accuracy
- [ ] 用原版 Crab 分别在**原始测试集**和**改写测试集**推理
- [ ] 收集并整理 4 组核心数据：

  | 数据集 | Head Acc | Tail Acc |
  |--------|----------|----------|
  | 原始题 |          |          |
  | 改写题 |          |          |

- [ ] 分析：Tail 明显低于 Head → 模型对模板化问题存在答题捷径，即 **C1 创新点的直接论文证据**

---

## Day 5（3/7）— C2 模型改造：AV-CrossAttn 模块实现

- [ ] 在 `models/multimodal_encoder.py` 中实现 `AVCrossAttentionFusion` 类：
  ```python
  class AVCrossAttentionFusion(nn.Module):
      def __init__(self, d_model=4096, nhead=8):
          super().__init__()
          self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
          self.norm = nn.LayerNorm(d_model)

      def forward(self, audio_feat, video_feat):
          # audio_feat: (B, 32, D)  video_feat: (B, 32, D)
          fused, _ = self.cross_attn(query=audio_feat, key=video_feat, value=video_feat)
          return self.norm(audio_feat + fused)
  ```
- [ ] 修改 `ALProjector.forward()` 接入 Cross-Attention
- [ ] 修改 `configs/unified_config.py`，新增 `use_av_crossattn: bool = False`
- [ ] 修改 `models/unified_arch.py` 的 `init_multimodal_modules()` 注册新模块
- [ ] 单元测试：构造随机 Tensor，验证维度正确、前向传播无报错

---

## Day 6（3/8）— C3 训练改造：MCCD 去偏损失实现

- [ ] 在 `models/unified_llama.py` 的 `forward()` 中新增 `forward_single_modal()` 辅助方法
- [ ] 实现 MCCD 去偏损失分支：
  ```python
  if self.training and use_mccd:
      v_logits = self.forward_single_modal(inputs_embeds, mask='audio')
      a_logits = self.forward_single_modal(inputs_embeds, mask='video')
      kl_loss = kl_div(P_av, P_v) + kl_div(P_av, P_a)
      total_loss = ce_loss + self.mccd_lambda * kl_loss
  ```
- [ ] 修改 `configs/unified_config.py`，新增 `use_mccd: bool = False`、`mccd_lambda: float = 0.1`
- [ ] 取前 100 条数据验证：loss 能正常反传，无 `NaN`

---

## Day 7（3/9）— 整合 + 论文写作启动

- [ ] 将 C2 + C3 改动合并，完整 forward 一次（不需要完整训练，只验证代码逻辑正确）
- [ ] 整理本周所有实验数据，汇总到表格
- [ ] 开始写**论文第3章草稿**：
  - 3.1 MUSIC-AVQA 偏置分析（Day 1、4 的数字）
  - 3.2 MUSIC-AVQA-R 数据集构建（C1）
  - 3.3 模型改进方案概述（C2、C3）

---

## 优先级速查

| 优先级 | 任务 | 说明 |
|--------|------|------|
| 🔴 必做 | 基线评估 + 音频消融 | 论文核心数据，无可替代 |
| 🔴 必做 | Head/Tail 划分 + 偏置分析 | C1 贡献的直接实验证据 |
| 🟡 重要 | 问题重写（C1） | 数据集贡献，可批量自动化 |
| 🟡 重要 | AV-CrossAttn 实现（C2） | 本周实现代码，下周再跑训练 |
| 🟢 次优 | MCCD 损失实现（C3） | 代码量小，先跑通逻辑即可 |
| 🟢 次优 | 论文第3章草稿 | 边做实验边写，不必等数据齐全 |
