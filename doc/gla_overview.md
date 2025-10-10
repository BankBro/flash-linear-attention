# Gated Linear Attention (GLA) 零基础速读

> 目标：花 10 分钟弄清楚 GLA 模型做什么、代码在哪、每一层负责什么，并能根据提示去源码定位。

---

## 1. 它是什么？解决什么问题？
- **GLA = Gated Linear Attention**：一种把标准自注意力变成线性复杂度的 Transformer 变体。长序列训练/推理时更快、更省显存。
- **工作流程**：输入 token → 词向量 → 重复堆叠的 `GLABlock`（线性注意力 + 门控前馈）→ 输出 logits。
- **工程特点**：大量 fused kernel（RMSNorm、SwiGLU、CrossEntropy 等）和缓存优化，亲和 GPU。

---

## 2. 文件地图（最常用的都在这里）

| 组件 | 作用 | 代码位置 |
| --- | --- | --- |
| `GLAForCausalLM` | Hugging Face 兼容的顶层模型（训练/推理接口） | `fla/models/gla/modeling_gla.py` |
| `GLAModel` | 主干网络，堆叠多个 `GLABlock` | `fla/models/gla/modeling_gla.py` |
| `GLABlock` | 单层 Transformer 结构（线性注意力 + GatedMLP） | `fla/models/gla/modeling_gla.py` |
| `GatedLinearAttention` | 线性注意力内核 | `fla/layers/gla.py` |
| `GatedMLP` & `swiglu_linear` | 门控前馈层 & fused SwiGLU | `fla/modules/mlp.py` |
| `GLAConfig` | 配置项定义 | `fla/models/gla/configuration_gla.py` |
| `Cache` | 推理时缓存 KV/卷积状态 | `fla/models/utils.py` |
| `legacy/training` | 旧版训练脚本与数据处理 | `legacy/training/` |

只要记住这张表，就能快速跳到需要的源码。

---

## 3. 数据流长什么样？

```
[Token IDs]
    │  (nn.Embedding)
    ▼
[GLABlock × N]  ← 每层包含线性注意力 + Gated MLP，两次残差连接
    │
    ▼  (RMSNorm)
[Hidden States]
    │  (Linear = lm_head)
    ▼
[Logits]
```

如果用 Mermaid 预览（如 https://mermaid.live），可以看到层级关系：
```mermaid
graph TD
    A[GLAForCausalLM<br/>(modeling_gla.py)] --> B[GLAModel]
    B --> B1[Embedding]
    B --> B2[GLABlock × num_hidden_layers]
    B --> B3[Final RMSNorm]
    A --> A1[LM Head]
    A --> A2[Loss / Fused Loss]
    B2 --> C1[Attention Branch<br/>(GLA or MHSA)]
    B2 --> C2[GatedMLP Branch]
    C1 --> D1[GatedLinearAttention<br/>(layers/gla.py)]
    C2 --> D2[swiglu_linear<br/>(modules/mlp.py)]
    A -.-> Cache[Cache<br/>(models/utils.py)]
```

---

## 4. 从外到内拆解核心类

### 4.1 `GLAForCausalLM`（顶层包装）
- **位置**：`fla/models/gla/modeling_gla.py`
- **职责**：
  - 持有 `GLAModel` 主体与 `lm_head` 输出层，封装成 Hugging Face 兼容 API。
  - 训练时根据配置选择普通或 fused 交叉熵；推理时处理 `past_key_values`。
  - 你要加载/保存/生成时的入口就是它。

### 4.2 `GLAModel`（主干网络）
- 按 `GLAConfig` 构造：`Embedding → GLABlock × N → RMSNorm`。
- `forward` 支持 `input_ids` 或 `inputs_embeds`，并处理可选的 `attention_mask`、`past_key_values`、`use_cache`。
- 训练模式下强制 `use_cache=False`，避免显存浪费；推理时可开启缓存加速。

### 4.3 `GLABlock`（单层结构）
- 结构完全类比 Transformer，但注意力换成了 GLA（可选回退到标准多头 Attention）。
- 两个子路径：
  1. `attn_norm` → `self.attn` → 残差加回。
  2. `mlp_norm` → `GatedMLP` → 残差加回。
- `fuse_norm=True` 时，RMSNorm 和残差在一个 fused kernel 里完成，GPU 上更省显存。

### 4.4 `GatedLinearAttention`（线性注意力核）
- **关键步骤**：
  1. 投影：`q_proj/k_proj/v_proj` 把隐藏状态映射到头维度。
  2. 特征映射：用 `config.feature_map` 指定的函数（如 `swish`）处理 Q/K。
  3. 门控：`gk_proj` + `logsigmoid` 生成门值，避免数值爆炸。
  4. 选择内核：根据 `attn_mode` 调用 `chunk_gla`（训练友好）、`fused_recurrent_gla`（推理友好）或 `fused_chunk_gla`。
  5. 可选短卷积：`use_short_conv=True` 时，先用 `ShortConvolution` 提前聚合局部上下文。
- **缓存配合**：推理时通过 `Cache` 保存 recurrent/卷积状态，实现增量计算。

### 4.5 `GatedMLP`（门控前馈网络）
- 三条线性支路：`gate_proj`、`up_proj`、`down_proj`。
- 默认 `fuse_swiglu=True`，调用 `swiglu_linear` fused kernel 一次性完成 SwiGLU + Linear，减少中间激活保存。
- 支持张量并行：`SwiGLULinearParallel` 可以指定输入/输出的张量布局。

### 4.6 配套模块
- **`RMSNorm` / `FusedRMSNormGated`**（`fla/modules/layernorm.py`）：
  - 支持 residual 融合、FP32 计算，配合 `fuse_norm` 开关使用。
- **`Cache`**（`fla/models/utils.py`）：
  - 扩展自 Transformers 的 cache，可以记录 GLA 每层的 recurrent / conv / attn 状态，支持窗口滚动和 legacy 格式互转。
- **`SimpleGatedLinearAttention`**（`fla/layers/simple_gla.py`）：
  - 门控按头共享的轻量版，适合快速实验或移动端部署。
- **注册入口**（`fla/models/gla/__init__.py`）：
  - 把配置与模型注册到 Hugging Face `Auto*`，调用时直接 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 即可。

---

## 5. `GLAConfig` 要怎么配？

`GLAConfig`（`fla/models/gla/configuration_gla.py`）负责声明每个部件的尺寸和行为。记住下面这些分类就行：

1. **模型规模**：`hidden_size`、`num_hidden_layers`、`num_heads`、`num_kv_heads`、`vocab_size`。
2. **注意力形态**：`attn_mode`（选择 kernel）、`feature_map`、`expand_k/v`、`use_short_conv`、`conv_size`、`use_output_gate`。
3. **混合注意力**：`attn` 字典，可指定某些层使用标准 Attention，并传入相关参数（`num_heads`、`rope_theta`、`window_size` 等）。
4. **前馈与激活**：`hidden_ratio`、`intermediate_size`（未指定会自动按比例向上取整到 256 的倍数）、`hidden_act`、`fuse_swiglu`。
5. **归一化与稳定性**：`fuse_norm`、`elementwise_affine`、`norm_eps`、`clamp_min`、`initializer_range`。
6. **训练/推理行为**：`use_cache`、`fuse_cross_entropy`、`pad_token_id` / `bos_token_id` / `eos_token_id`、`tie_word_embeddings`。

这些字段直接传入构造函数即可，例如：
```python
from fla.models.gla import GLAConfig, GLAForCausalLM

config = GLAConfig(
    num_hidden_layers=12,
    hidden_size=1024,
    attn_mode="fused_chunk",
    feature_map="swish",
    use_short_conv=True,
)
model = GLAForCausalLM(config)
```

---

## 6. 注意力公式长什么样？

\[
\text{GLA}(Q, K, V) = \sigma(G) \odot \left(\phi(Q) \cdot \left(\phi(K)^\top V\right)\right)
\]
- \(Q, K, V\)： Projection 后的张量。
- \(\phi(\cdot)\)：`feature_map` 指定的特征映射（如 `swish`）。
- \(G\)：门控投影的输出，`logsigmoid` 让其数值更稳定。
- 计算上通过 prefix-scan / chunk 技巧把注意力复杂度降到线性。

想看具体实现，请追踪到 `fla/layers/gla.py` 中 `GatedLinearAttention.forward`，再往里看 `chunk_gla` / `fused_*` 内核。

---

## 7. 训练 & 推理常见问题

### 7.1 训练阶段
- 默认不使用 `attention_mask`（旧版数据脚本会把文本拼成定长块），`use_cache` 会被自动关掉。
- 启用梯度检查点（`model.gradient_checkpointing_enable()`）时，框架会提示并强制关闭缓存。
- Fused cross entropy 只有在训练 + 提供 `labels` 的情况下才会启用。

### 7.2 推理 / 增量生成
- `prepare_inputs_for_generation` 只在第一步使用全部输入，之后每一步只传最新 token。
- `past_key_values` 使用项目里的 `Cache` 类，内部存有 GLA 的 recurrent/卷积状态。
- 序列较短时，注意力内核自动切换为 `fused_recurrent`，以减少 kernel 启动开销。

---

## 8. 数据处理 & attention_mask
- 老的训练 pipeline 在 `legacy/training/` 中：
  - `HuggingfaceDataset` 把语料拼成长缓冲区，切成固定长度。
  - Tokenizer 调用写死 `return_attention_mask=False`，所以默认没有 mask。
  - Collator 会把 padding 的 label 改成 `-100`，保证不会计算损失。
- 如果你需要 mask，只要改 tokenizer 调用为 `return_attention_mask=True`，GLA 层会自动先做 unpad，再在有效 token 上计算。

---

## 9. 小白也能用的调参提示
1. **混合注意力**：在 `attn["layers"]` 中填入层号，把部分层换回标准 Attention，以获得更强的全局建模能力。
2. **feature_map**：`swish` 是一个折中选择，`relu` 更稀疏、`gelu` 更平滑，根据任务试试就知道。
3. **短卷积**：本质是序列上的小卷积，适合语音、代码等强调局部模式的任务；同时要关注额外的缓存开销。
4. **fuse_norm / fuse_swiglu**：GPU 训练时强烈建议开启，显存友好；若要在 CPU 或调试环境运行，可以关掉以方便排查。
5. **fuse_cross_entropy**：默认开启，若需要查看 logits 或自定义 loss，可先关掉再覆写。

---

## 10. 参考资料 & 下一步
- 论文：[Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
- 一次性阅读顺序建议：`modeling_gla.py` → `gla.py` → `mlp.py` → `layernorm.py`
- 如果要自己训练，参考 `legacy/training/run.py` 看如何构建数据集与 Trainer。

走到这里，你已经掌握了 GLA 的整体脉络。接下来可以尝试：
1. 修改 `GLAConfig` 里的超参，看训练日志和显存占用怎么变。
2. 在某些层启用混合注意力，比较收敛速度。
3. 用自己的语料测试增量生成，感受 `Cache` 的速度提升。

祝你实验顺利，玩的开心！
