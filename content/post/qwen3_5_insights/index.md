---
title: "Qwen3.5 观测记录"
description: "总之是碰到什么学什么"
date: 2026-06-04T11:41:11+08:00
image: 9895831_p0.jpg
math: true
hidden: false
comments: true
draft: false
weight: 1
---

最近主播想搓一个完整的推理 pipeline 玩，看来看去最后选了 qwen3.5

此处为搓的过程中记的笔记。前半部分都是相关理论的理解和式子推导，后半部分是捋 transformers 代码实现的笔记。总之是碰到什么学什么，都记到一块了

感觉有点像什么实验报告

---

去除掉量化版本和 base model，Qwen3.5 系列的模型主要有这几个：

```
sparse - MoE:
  Qwen3.5-397B-A17B
  Qwen3.5-122B-A10B
  Qwen3.5-35B-A3B

dense:
  Qwen3.5-27B
  Qwen3.5-9B
  Qwen3.5-4B
  Qwen3.5-2B
  Qwen3.5-0.8B
```

---

## Gated DeltaNet

```
in Multi-Head Attention:
[B, L, H, D] -> [Batch, Seq_length, Num_heads, Head_dim]
```

https://blog.sunlingzhang.com/2026/03/05/Paper/Gated_DeltaNet/index.html

https://sustcsonglin.github.io/blog/2024/deltanet-1/

### softmax attn

回顾经典 softmax attention 算法

$$
O = \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} \odot M\right)V
$$

在这里不妨假设是单头注意力机制，同时也忽略 batch 维度。$M$ 为 causal mask

设 $L$ 为序列长度，$d$ 为 embedding 维度

从形状来说经历这样的过程：

- $Q$ `: [L, d]`，$K$ `: [L, d]`，$K^\top$ `: [d, L]`，$V$ `: [L, d]`
- $S = QK^\top$ `: [L, L]`
- $S' = \text{softmax}(S \odot M/ \sqrt{d})$ `: [L, L]`
- $O = S'V$ `: [L, d]`

我们将 $Q$、$K$、$V$、$O$ 按行分块，也就是将每行视为一个元素

<u>用 $\v q_t$ 表示 $Q$ 的第 $t$ 行元素对应的**列向量**</u>，则有

- $S_{ij} = \v q_i^\top \v k_j$，$j\le i$
- $S'_{ij} = \dfrac{\exp S_{ij}}{\sum \exp S_{ik}} / c$，$j\le i$
- $\v o_i =\v s_i'^\top V$

将结果逐层展开，可以得到

$$
\begin{align*}
\v o_t &= \v s_t'^\top V \\
&= \sum_{j=1}^t S'_{tj} \cdot \v v_j \\
&= \sum_{j=1}^t \frac{\exp S_{tj}}{\sum_{l=1}^t \exp S_{tl}} \v v_j \\
&= \sum_{j=1}^t \frac{\exp (\v q_t^\top \v k_j)}{\sum_{l=1}^t \exp (\v q_t^\top \v k_l)} \v v_j
\end{align*}
$$

这就是 Iterative Inference 的计算形式。其中省略了除以 $\sqrt{d}$ 的部分

在推理时，每次新计算一个 token，都要将 $\v q_t$ 和之前所有的 $\v k_{j\le t}$ 做点积。随着上下文增长，kv cache 和计算复杂度都会增长

### linear attn

为了尝试降低复杂度，我们必须想办法拆式子以调整计算顺序，其核心就是尝试核方法来干掉 softmax，从而获得计算的线性性，进而调整式子

具体来说，我们是要近似

$$
\exp (\v q^\top \v k) \rightarrow \text{sim}(\v q,\v k) = \phi(\v q)^\top \phi(\v k)
$$

这里假设 $\v q$ 是 `[d, 1]` 的列向量，$\phi(\v q)$ 是 `[m, 1]` 的列向量

之后式子变成

$$
\begin{align*}
\v o_t &= \sum_{j=1}^t \frac{\phi(\v q_t)^\top \phi(\v k_j)}{\sum_{l=1}^t \phi(\v q_t)^\top \phi(\v k_l)} \v v_j \\
&= \frac{\sum_{i\le t} \big(\phi(\v q_t)^\top \phi(\v k_i)\big) \v v_i}{\sum_{i\le t} \phi(\v q_t)^\top \phi(\v k_i)} \\
&= \frac{\big(\sum_{i\le t}\v v_i \phi(\v k_i)^\top\big) \phi(\v q_t)}{\big(\sum_{i\le t} \phi(\v k_i)^\top\big) \phi(\v q_t)}
\end{align*}
$$

取

$$
\begin{align*}
S_t &= \sum_{i\le t} \v v_i\phi(\v k_i)^\top \\
z_t &= \sum_{i\le t} \phi(\v k_i)^\top
\end{align*}
$$

其中 $S_t$ `: [d, m]`，$z_t$ `: [1, m]`，则有

$$
\v o_t = \frac{S_t \phi(\v q_t)}{z_t \phi(\v q_t)}
$$

每推理一个 token，进行这样的更新：

$$
\begin{align*}
S_t &= S_{t-1} + \v v_i\phi(\v k_i)^\top \\
z_t &= z_{t - 1} + \phi(\v k_i)^\top
\end{align*}
$$

观察 $S_t$ 的更新过程，可以看到它类似于一个隐藏状态为矩阵的 RNN。更准确地说，$S_t$ 实际上是 **key-value 外积的联想记忆矩阵**

联想记忆就是说，我们每次更新都向这个矩阵叠加一个 $\v v_i\phi(\v k_i)^\top$，在查询时：

$$
S_t\phi(\v q_t) = \sum_{i} \big(\v v_i\phi(\v k_i)^\top\big) \phi(\v q_t) = \sum_i \big(\phi(\v q_i)^\top \phi(\v k_i) \big)\v v_i
$$

也就是当我们展回到一开始拆 linear attention 的模式，它相当于将 $\v q$ 与 $\v k$ 内积之后对 $\v v$ 加权得到的结果

但是问题在于，随着累积的 kv 越来越多，因为 key 之间并不完全相互正交，在查询一个 key 时受到的干扰项就会更多，发生记忆冲突。当累积 kv 数量超过 feature 维度大小时冲突会加剧，因为 key 挤满了所有维度之后更不容易正交

为了避免重蹈 RNN 的覆辙，我们需要做出改变。我们不希望所有的 kv 都持续且平等地累加到这个线性记忆池中，而是要引入 CRUD 能力来维护这个记忆池

### DeltaNet

引入 delta rule 的目的是用增量更新代替直接累加 kv 外积

假设当前的状态矩阵为 $S_{t-1}$，给定当前 key $\v k_t$，我们首先进行一次读取

$$
\hat{\v v}_t = S_{t-1}\v k_t
$$

计算我们期望的结果与当前读取结果的偏差

$$
\delta_t = \v v_t - \hat{\v v}_t
$$

用这个 $\delta$ 与 $\v k_t$ 做外积更新状态矩阵

$$
S_t = S_{t-1}+\delta_t k_t^\top
$$

总结来说，我们的状态矩阵更新应该这样：

$$
S_t = S_{t-1} + (\v v_t - S_{t-1}\v k_t)\v k_t^\top
$$

DeltaNet 在上下文检索中表现出了很强的能力。事实上，delta rule 的更新规则实际上相当于在每一步最小化 MSE，这可以成为理解的一个角度

从这里开始现代做法已经不需要再借助核函数了，输出也简化为了

$$
\v o_t = S_t\v q_t
$$

这一方法虽然实现了有效率地更新，但是更新过程仍然是机械的，换句话说，每一步的更新强度相同。但是在实际序列建模中，不同 token 应该产生的更新规模差异很大

为此我们需要再添加门控，以便让模型自适应地调整记忆更新的强度

### Gated DeltaNet

在 DeltaNet 的基础上，我们一般能添加两个门控

$$
S_t = f_t \odot S_{t-1} + g_t\cdot (\v v_t - S_{t-1}\phi(\v k_t))\phi(\v k_t)^\top
$$

其中 $g_t$ 是**写入门**，$f_t$ 是**遗忘门**。这两个门的形状不固定，一般可以假设 $f_t$ 是可广播到 $S$ 上的向量，$g_t$ 是标量或者可广播到 $S$ 的向量

$f_t$、$g_t$ 的定义给出了这两个门的可学习参数。例如我们可以设定

$$
\v f_t = \sigma(W_f\v x_t + \v b_f) \\
\v g_t = \sigma(W_g\v x_t + \v b_g)
$$

其中 $\v x_t$ 是当前层的输入，$W_f$、$\v b_f$、$W_g$、$\v b_g$ 就是门的可学习参数，它们描述了「对于某个 token，应该施加怎样的学习强度」

如果是多头 attention，也可以在每一个头上都设置一个门，附带相应的可学习参数

---

## RoPE

**Rotary Position Embedding**，一种通过旋转的方式进行位置编码的算法。我们可以简要推导它来理解算法原理

https://lorenzocesconetto.github.io/

https://arxiv.org/abs/2104.09864

---

我们的目标是构造一种更有效的位置编码方式

Attention is All You Need 论文里使用的位置编码方式是基于一个 closed form equation 生成位置信息，然后直接加到 hidden states 上，早期的其他做法还包括引入可学习的位置向量加到 hidden states 上

但是事实证明，在 hidden states 上直接加位置信息是一个坏主意。这种把绝对位置编码和 token 语义信息混在一起的做法极易导致对绝对位置的过拟合学习，模型实际上在记忆绝对位置而非学习相对关系

模型的上下文长度也受到训练数据限制，使用 closed form equation 编码的模型在输出超过最长训练上下文长度之后性能会严重下降，因为模型以前并没有记忆过超出的这些位置；使用可学习位置向量的模型输出长度更是不能超过最长的训练上下文，否则就没有位置向量可用了

因此我们必须在特定的时刻再附加上位置编码，准确来说，应该在 attention 计算中，给 QK 矩阵上加位置编码，这样限定了模型在 token 之间 attention 时考虑其位置

其次，closed form equation 是可取的，设计这个确定性函数是我们引导模型学习位置关系的最好方法。并且这个函数必须满足两个 token 的向量经过这个函数之后在点积结果下产生的额外因子与且仅与两个 token 的位置差值相关

---

总结下来，我们的位置编码方案需要满足以下几项

- 在 full attention 计算中给 QK 添加位置编码
- 使用确定性函数生成位置编码
- 原来的 QK 在经过该确定性函数之后，一对 token 向量的点积结果中多出来的项或因子与且仅与这两个 token 的位置差值相关

---

设我们要寻找一个变换 $f(\v x, m)$，使得对于任意 query $\v q$ 和 key $\v k$，都有

$$
\langle f(\v q, m), f(\v k, n)\rangle = g(\v q, \v k, m-n)
$$

我们还假定 $f$ 是线性变换，也就是输入乘上矩阵 $R_m$，并且 $f$ 还要是正交变换，为了最大程度保留原始向量的语义

可以推导出 $R_m$ 必须满足

$$
R_m^\top R_n = R_{n-m}
$$

所以问题是，在 $d$ 维空间里，哪些正交矩阵族 $\{R_m\}$ 满足 $R_m^\top R_n = R_{n-m}$ 

在 2d 空间里，答案就是旋转矩阵，即取 $R_m = \begin{bmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{bmatrix}$

>在 3d 或更高维空间中，一般的旋转群是**非交换**的。如果 $R_m$ 是任意三维旋转，$R_m^\top R_n$ 不会简单地等于某个只由 $n-m$ 决定的旋转矩阵，除非所有旋转都围绕**同一个固定轴**。而如果都围绕同一个固定轴旋转，那么这个变换实际上只作用在一个 2d 平面上（垂直于轴的平面），剩余维度保持不变，本质上还是“二维旋转 + 一维恒等”的组合

> 群表示论告诉我们：实数域上正交群的一维不可约表示是 ±1（反射），二维不可约表示正是旋转矩阵 SO(2)。任何满足群同态性质的高维正交表示，都可以**分解为一堆二维旋转块（和可能的一维恒等块）的直和**

因为符合要求的高维向量正交线性变换都能拆分成若干二维旋转，因此我们采用的计算方式如下：

对于 query 和 key 的每一个 token 的向量，假设 $\v q_t$`: [d]`，d 是偶数，我们从头开始不断地取向量的两维为一组，把他们看成一个 2d 向量，然后应用一个 2*2 的矩阵将它旋转

即设

$$
\v q_t = (x_0, x_1,x_2,x_3,\dots)
$$

则令

$$
f(\v q_t, p) = \left(R’_{p0} \begin{bmatrix} x_0 \\ x_1\end{bmatrix}, R’_{p1} \begin{bmatrix} x_2 \\ x_3\end{bmatrix},\dots,R’_{pi} \begin{bmatrix} x_{2i} \\ x_{2i+1}\end{bmatrix},\dots\right)
$$

其中 $R'_{pi}$ 是一个和 $p$ 与 $i$ 相关的二维旋转矩阵

$$
R'_{pi} = \begin{bmatrix}\cos p\theta_i & -\sin p\theta_i \\
\sin p\theta_i & \cos p\theta_i\end{bmatrix}
$$

其中 $\theta_i$ 是角频率，它控制在不同维度上旋转的快慢

这一整个操作相当于对原来的 $\v q_t$ 乘上了这样一个旋转矩阵

$$
\begin{bmatrix} R'_{p0} \\
& R'_{p1} \\
& & R'_{p2} \\
&&&\ddots \\
&&&& R'_{pd/2}\end{bmatrix}
$$

这就是 RoPE 的核心操作

我们一般取 $\theta_i$ 为

$$
\theta_i = 10000^{-2i/d}
$$

这里 i 越小，$\theta_i$ 越大。让低维度旋转更快是约定俗成做法，继承自传统 transformer 的正弦编码，保持低维度高频，高维度低频

---

回顾我们的出发点假设，现在构造的这个变换满足 $\langle f(\v q, m), f(\v k, n)\rangle = g(\v q, \v k, m-n)$，我们可以计算出这个式子来查看 $g$ 的构成

取做过变换后的向量 query $\v q' = f(\v q,m)$ 和 key $\v k' = f(\v k, n)$ 进行点积

$$
\begin{align*}
\v q'^\top \v k' & = \sum_{i=0}^{d/2} \left(R'_{mi}\begin{bmatrix}\v q_{2i} \\ \v q_{2i+1}\end{bmatrix}\right)^\top \left(R'_{ni}\begin{bmatrix}\v k_{2i} \\ \v k_{2i+1}\end{bmatrix}\right)\\
&= \sum_{i=0}^{d/2} \begin{bmatrix}\v q_{2i} \\ \v q_{2i+1}\end{bmatrix}^\top R'^\top_{mi} R'_{ni} \begin{bmatrix}\v k_{2i} \\ \v k_{2i+1}\end{bmatrix}\\
&= \sum_{i=0}^{d/2} \begin{bmatrix}\v q_{2i} \\ \v q_{2i+1}\end{bmatrix}^\top R'_{\Delta i} \begin{bmatrix}\v k_{2i} \\ \v k_{2i+1}\end{bmatrix}
\quad\quad\quad\quad \Delta=n-m,\quad R'^\top_{m} R'_n = R'_{n-m} \\
&= \sum_{i=0}^{d/2} \v q_{2i}(\v k_{2i}\cos\Delta\theta_i - \v k_{2i+1}\sin\Delta\theta_i) + \v q_{2i+1}(\v k_{2i}\sin\Delta\theta_i+\v k_{2i+1}\cos\Delta\theta_i)\\
&= \sum_{i=0}^{d/2} (\v q_{2i}\v k_{2i} + \v q_{2i+1}\v k_{2i+1})\cos\Delta\theta_i + (\v q_{2i}\v k_{2i+1}-\v q_{2i+1}\v k_{2i})\sin\Delta\theta_i
\end{align*}
$$

可以发现，$\v q_{2i}\v k_{2i} + \v q_{2i+1}\v k_{2i+1}$ 是这两维原来的点积，而 $\v q_{2i}\v k_{2i+1}-\v q_{2i+1}\v k_{2i}$ 是这两维的叉积

---

综合上述内容，RoPE 本质上是将 QK 的每一个 token 的向量都划分为了 $d/2$ 个二维平面，在其上施加一个旋转量。经过此操作之后的 qk 点积保留了原来的点积和差积项，同时添加了只与两个 token 的位置差值相关的因子

因此我们可以说，<u>RoPE 改变了 token 比较时的几何关系，调制了相似度函数使其包含了相对位置的语义</u>

---

## Architecture

https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json

https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5_moe

我们主要从 hf 上的 config.json 和 transformers 库的 `qwen3_5_moe` 实现来了解具体 pipeline

从 `src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py` 中可以看到所有的类

对于 text generation 来说，入口在 `Qwen3_5MoeForCausalLM.forward`

forward 函数已经注明了该类的使用方法

```python
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

model = Qwen3_5MoeForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, 
    skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

对于 `Qwen3_5MoeForCausalLM` 类，我们在 `__init__` 里引入 `Qwen3_5MoeTextModel` 类，并在 `forward` 里调用它获得输出

```python
    def __init__(self, config):
        self.model = Qwen3_5MoeTextModel(config)
    
    def forward(
    ) -> MoeCausalLMOutputWithPast:
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            **kwargs,
        )
```

对于 `Qwen3_5MoeTextModel` 类，我们在 `__init__` 里初始化所有的 layers，在 `forward` 里将 hidden_states 依次通过这些 layer，得到结果

```python
    def __init__(self, config: Qwen3_5MoeTextConfig):
        self.layers = nn.ModuleList(
            [Qwen3_5MoeDecoderLayer(config, layer_idx) 
             for layer_idx in range(config.num_hidden_layers)]
        )
    
    def forward(
    ) -> BaseModelOutputWithPast:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if self.config.layer_types[i] == 
            	"linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
```

因此实际执行计算的是 `Qwen3_5MoeDecoderLayer`，它在 `forward` 里执行如下运算：

- input_layernorm
- linear_attention 或者 full_attention
- hidden_states 和一开始的 hidden_states 进行一次残差连接
- post_attention_layernorm
- mlp
- hidden_states 和上一次残差连接之后、post_attention_layernorm 之前的 hidden_states 做一次残差连接

在这里会根据当前的 layer_type 选择 attn 实例为 `Qwen3_5MoeGatedDeltaNet` 或者 `Qwen3_5MoeAttention`

input_layernorm 和 post_attention_layernorm 用的都是 `Qwen3_5MoeRMSNorm`

mlp 部分使用 `Qwen3_5MoeSparseMoeBlock`

---

在 class 定义中我们会声明很多 nn.Module 成员，其中包含着可学习参数

在启动模型推理时，通过 `.from_pretrained` 方法，我们整个路径上的 class 实例化，接着从数据源的 checkpoint 读取训练好的参数，加载到具体的 nn.Module 里的 tensor

因此在 class 定义里不包含读取数据的代码，只有 `__init__` 中的声明和 `forward` 里的使用

---

### Qwen3_5MoeGatedDeltaNet

在 `forward` 中，hidden_states 首先会应用 padding mask

```python
hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
```

当 batch 中存在若干个长短不一的 token 序列时就需要 padding，在短的 token 序列的末尾补上 `<PAD>` token，将所有序列长度对齐最长的 seq_len

但是在计算 attention 时，我们不希望这个 `<PAD>` token 产生任何注意力分数贡献，因此就需要一个 attention mask，这个 mask 标注了哪些部分是真实 token，哪些部分是 padding

```
input_ids:      [今, 天, 天, 气, 真, 好, PAD, PAD, PAD, PAD, PAD, PAD]
attention_mask: [ 1,  1,  1,  1,  1,  1,  0,   0,   0,   0,   0,   0]
```

对于经典 softmax attention 来说，我们可以在计算完 $QK^\top$ 之后把 attention mask 连同 causal mask 一起应用，将不需要的位置置为 `-inf`。因为 $QK^\top$ 显式地计算出了每一个 token 作为 query 和每一个 token 作为 key 的查询值，因此我们可以直接将结果矩阵中所有 `<PAD>` 作为 key 受查询的列去掉

但是在 GDN 里，我们将先计算出 kv 外积 $\v v \v k^\top$，累加到记忆矩阵里，最后用一个 query 查询整个记忆矩阵。因此我们必须提前到在 hidden_states 产生 qkv 之前，就把其中的 `<PAD>` 置为 0，这样才能消除 padding 的影响

```python
def apply_mask_to_padding_states(hidden_states, attention_mask):
    # NOTE: attention mask is a 2D boolean tensor
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states
```

在 `apply_mask_to_padding_states` 函数中存在条件 `shape[1] > 1 and shape[0] > 1`，表示只有当前 hidden_states 的 batch_size > 1，并且最长 token 序列长度大于 1 的时候才考虑使用 attention mask。这表明至少有一个序列在本轮推理中在做 prefill，因为在 decode 阶段每个序列每步都只有一个 token

---

忽略一些 cache 处理，接下来我们会从 hidden_states 产生 mixed_qkv

设当前 hidden_states 的形状是 `[B, L, D]`，那么要产生的 mixed_qkv 形状为 `[B, L, Q_dim + K_dim + V_dim]`，也就是 QKV 连起来的一个矩阵

产生 mixed_qkv 的目的是在之后进行一次 causal conv1d，conv1d 的作用是将每个 token 换成它及它之前几个 token 的 embedding 向量的加权和。它的作用是先给每个 token 注入一些局部上下文，以加强线性注意力的局部建模能力

```python
mixed_qkv = self.in_proj_qkv(hidden_states)
mixed_qkv = mixed_qkv.transpose(1, 2)
```

self.in_proj_qkv 是一个 `nn.Linear` 模块。而 `key_dim * 2 + value_dim` 就是 QKV 的 dim 之和，它也被定义为了 `conv_dim`

```python
self.in_proj_qkv = nn.Linear(self.hidden_size, 
                             self.key_dim * 2 + self.value_dim, bias=False)
self.conv_dim = self.key_dim * 2 + self.value_dim
```

进行 `transpose` 的目的把原来的 `[B, L, conv_dim]` 改成 `[B, conv_dim, L]`，对齐 conv1d 中的 channel=conv_dim，length=seq_len

---

从 hidden_states 还会产生 z, a, b

```python
z = self.in_proj_z(hidden_states)  # Shape: (batch, seq_len, value_dim)
z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
# 最后用于 gated normalization
core_attn_out = self.norm(core_attn_out, z)  # Line 541

b = self.in_proj_b(hidden_states)  # Shape: (batch, seq_len, num_v_heads)
beta = b.sigmoid()                  # Line 503

a = self.in_proj_a(hidden_states)  # Shape: (batch, seq_len, num_v_heads)
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # Line 505
```

z 是最后执行 Gated RMSNorm 的门控向量，b 是写入强度中间量，a 是用于计算衰减因子的中间量

---

接下来执行 causal conv1d

```python
mixed_qkv = self.causal_conv1d_fn(
	x=mixed_qkv,
	weight=self.conv1d.weight.squeeze(1),
	bias=self.conv1d.bias,
	activation=self.activation,
	seq_idx=None,
)
```

执行完成的同时会对每个元素运行激活函数，这里使用的是 SiLU

$$
\text{SiLU}(x) = x\cdot\text{sigmoid}(x)
$$

---

之后我们会把 mixed_qkv 转置回来，再切开来变成 qkv 矩阵

```python
mixed_qkv = mixed_qkv.transpose(1, 2)
query, key, value = torch.split(
    mixed_qkv,
    [
        self.key_dim,
        self.key_dim,
        self.value_dim,
    ],
    dim=-1,
)
```

切开之后还会分出 head，通过将最后一维 reshape 为单个 head_dim 维实现

```python
query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
```

---

之后我们会计算出 beta 和 g，这两个分别是真正的写入强度和衰减因子

```python
beta = b.sigmoid()
# If the model is loaded in fp16, without the .float() here, A might be -inf
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

这里 `F` 其实是 `import torch.nn.functional as F`

softplus 是一个平滑版的 ReLU，在 x << 0 时 softplus(x) 趋向于 0，函数整体是光滑的
$$
\text{softplus}(x) = \log(1+\exp(x))
$$
这里可以看到我们取出 A_log 做 exp 之后取负，A_log 实际就是 log 之后的 A 矩阵。在读取训练数据时我们能读到的是 A_log 而不是 A 的值

这样处理其实是为了数值稳定性，A 表示**基础衰减率**，必须是正数，如果直接训练 A 的话，梯度更新可能让 A 变成负数。因此我们改为训练 log(A)，在使用时使用 exp(log(A)) 恢复，这样 A 的值总是正数

`dt_bias` 是可学习参数，用来让模型能够微调衰减行为

综上，beta 和 g 因子的计算路径整体来看是这样的：

$$
\text{hidden\_state}\xrightarrow[]{\text{in\_proj\_b}} b \xrightarrow[]{\text{sigmoid}} \beta \\
\text{hidden\_state}\xrightarrow[]{\text{in\_proj\_a}} a \xrightarrow[]{+\mathrm{dt\_bias}}\xrightarrow[]{\text{softplus}}\xrightarrow{\times -A} g
$$

这里的 beta 因为经过了 sigmoid，每个值都在 0~1 之间，而 g 保证了每个值都是负数

在使用时，beta 会直接乘到 delta 上，作为写入强度

```python
delta = (v_t - kv_mem) * beta_t
```

而 g 则先计算 exp(g) 之后再乘到状态上。因为 g 是负数，exp(g) 总是在 0~1 之间，保证了衰减

```python
g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
last_recurrent_state = last_recurrent_state * g_t
```

---

如果 v 的头数多于 qk，那么我们会复制 qk 的头，使其和 v 的数量相等

```python
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
```

这里我们假设了如果 num_v_heads > num_k_heads，那么 num_v_heads 一定是 num_k_heads 的倍数，例如默认配置

```
  - linear_num_key_heads = 16
  - linear_num_value_heads = 32
```

之前我们已将 qkv 切分开，例如

```python
query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
```

其中 dim=2 的维就是 head index 维，沿 head 维复制，相当于我们让几个 value 头共享了相同的 qk

---

之后就是调用 kernel 计算 GDN

```python
if use_precomputed_states and seq_len == 1:
    core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
    )

else:
    core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=recurrent_state if use_precomputed_states else None,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
    )
```

对于 prefill 阶段，使用 `chunk_gated_delta_rule`，对于单个 token 的 decode，使用 `recurrent_gated_delta_rule`

算完 attention 之后，我们会先把输出拍成 2d 矩阵，z 也拍成 2d 矩阵

```python
core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
z = z.reshape(-1, self.head_v_dim)
```

z 是之前从 hidden_state 线性变换得出的量

```python
z = self.in_proj_z(hidden_states)
z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
```

我们会用 attention 结果和 z 做一个 RMSNormGated

```python
core_attn_out = self.norm(core_attn_out, z)
```

RMSNormGated 就是先算一下 RMSNorm，再乘上一个 SiLU(z) 即可

然后 attention 结果就可以恢复 `[B, L, D]` 的形状，最后通过一个输出线性变换之后返回即可

```python
core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
output = self.out_proj(core_attn_out)
return output
```

---

### Qwen3_5MoeAttention

这个 class 对应 full_attention layer，依旧从 `forward` 入手

开头处保存了一下相关的 shape

```python
input_shape = hidden_states.shape[:-1]
hidden_shape = (*input_shape, -1, self.head_dim)
```

hidden_states 的形状是 `[batch, seq_len, hidden_size]`，取 `[:-1]` 得到 `[batch, seq_len]`

hidden_shape 的形状是 `[batch, seq_len, -1, head_dim]`，也就是提前处理出分 head 之后的张量形状

---

接下来将 hidden_states 做线性变换，拆出来 q_states 和 gate

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2),
    2,
    dim=-1
)
gate = gate.reshape(*input_shape, -1)
```

首先 `self.q_proj(hidden_states)` 把隐藏状态投影成了 `[batch, seq_len, num_heads * head_dim * 2]` 的张量，注意最后一维有一个 *2

接着通过 `.view(*input_shape, -1, self.head_dim * 2)` 变成 `[batch, seq_len, num_heads, head_dim * 2]`

最后通过 `torch.chunk(..., 2, dim=-1)`，把整个张量沿 `dim=-1` 切成 `N=2` 份，得到 query_states 和 gate

最后把 gate 展平。最终结果是 query_states 形状为 `[batch, seq_len, num_heads, head_dim]`，gate 的形状为 `[batch, seq_len, num_heads * head_dim]`

---

然后我们会具体计算出 QKV

```python
query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

QKV 在从 hidden_states 线性变换过来之后都 reshape 到 hidden_shape 的 `[batch, seq_len, num_heads, head_dim]`（对于 query_states 来说其实不需要再 reshape 了）

同时 QK 这两个还做了一个 RMSNorm，对每个头的每个 token 在 head_dim 维上进行一次 norm

主要目的是为了保证数值稳定性，平整分布，同时去掉 QK 中嵌入向量的模长信息，只保留方向信息，再让他们做语义匹配，这样能少许提升模型的长上下文能力

嵌入向量的模长也会编码一部分语义，因此 V 作为最终被加权求和的向量不进行 norm

QKV 都做了 `.transpose(1, 2)`，他们最后的形状都是 `[batch, num_heads, seq_len, head_dim]`

---

然后我们会对 QK 做 RoPE 编码，RoPE 的原理可以见前文笔记

```python
cos, sin = position_embeddings
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
```

这里的 position_embeddings 的 cos, sin 是 forward 的参数，它在 Qwen3_5MoeTextModel 执行 forward 时调用 layer 用到，并且是在循环之前提前计算的

```python
hidden_states = inputs_embeds
position_embeddings = self.rotary_emb(hidden_states, position_ids)

for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
    layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask

    hidden_states = decoder_layer(
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=layer_mask,
        position_ids=text_position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        **kwargs,
    )
```

这里的 `self.rotary_emb` 在 `__init__` 里定义

```python
self.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=config)
```

这个模块我们之后再深入

---

忽略一点处理 kv cache 的部分。再之后就是取 attention 函数计算 attn。同时添加来自 gate 的门控

```python
attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
    self.config._attn_implementation, eager_attention_forward
)

attn_output, attn_weights = attention_interface(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0 if not self.training else self.attention_dropout,
    scaling=self.scaling,
    **kwargs,
)

attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = attn_output * torch.sigmoid(gate)
```

这里的 attn_weights 是指 $\text{softmax}(QK^\top)\times \text{scaling}$ 的结果，似乎对主线推理没什么用，可用于其他任务

最后我们把 output 进行一次 O 变换，返回即可

```python
attn_output = self.o_proj(attn_output)
return attn_output, attn_weights
```

---

### Qwen3_5MoeSparseMoeBlock

上面的类都是 attention 模块。在 Qwen3_5MoeDecoderLayer 中，我们会根据 config 里的 layer_type 来选择一种 attention，再搭配一个 mlp 模块。其中 mlp 使用的就是固定的 Qwen3_5MoeSparseMoeBlock

这里的结构是，hidden_states 会经过 gate 选择 top k 个 expert 做 moe，同时也会经过一个固定的 shared_expert，shared_expert 有自己专门的门控

最后 moe 的 output 会和 shared_expert 的输出一起相加之后返回

```python
batch_size, sequence_length, hidden_dim = hidden_states.shape
hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
shared_expert_output = self.shared_expert(hidden_states_reshaped)
_, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
expert_output = self.experts(hidden_states_reshaped, 
                             selected_experts, routing_weights)

shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) 
						* shared_expert_output

expert_output = expert_output + shared_expert_output
expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
return expert_output
```

其中涉及这几个组件

```python
self.gate = Qwen3_5MoeTopKRouter(config)
self.experts = Qwen3_5MoeExperts(config)
self.shared_expert = Qwen3_5MoeMLP(config, 
                     intermediate_size=config.shared_expert_intermediate_size)
```

#### Qwen3_5MoeTopKRouter

主要就是一个 hidden_dim 到 num_experts 的 linear 变换，最后返回 router_logits, router_scores, router_indices。不过 router_logits 在推理中被丢弃了

#### Qwen3_5MoeExperts

这里的情况要更复杂一点

通过 Qwen3_5MoeTopKRouter 我们获得了每个 token 选中的 top k 个 expert 以及相应的权重

我们会先找出所有被用到的 expert，然后统计 hidden_states 里有哪些 token 用到了这个 exprt，把他们的 hidden_states 里的向量都抽出来组成一个 current_hidden_states

current_hidden_states 首先经过 self.gate_up_proj，再 chunk 2 得到一个 gate 和一个 up

```python
gate, up = nn.functional.linear(current_state, 
                                self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
```

这两个东西构成了 SwiGLU 结构，不同于普通的 FFN，SwiGLU 会计算出两路线性变换结果，把其中一路作为门控

$$
\text{SwiGLU}(x) = W_2(\text{SiLU}(W_a x)\odot(W_bx))
$$

在这里 gate 对应 $W_ax$，up 对应 $W_b x$

根据上面的式子，接下来我们对 gate 做 SiLU，然后逐元素乘上 up

```python
current_hidden_states = self.act_fn(gate) * up
```

再做一个线性变换得到结果，self.down_proj 对应了 $W_2$

```python
current_hidden_states = nn.functional.linear(current_hidden_states, 
                                             self.down_proj[expert_idx])
```

然后把这个结果中的每个 token 乘上相应的 weight，拆包累加到 final_hidden_states 中。这个 final_hidden_states 一开始被初始化为了全 0

```python
current_hidden_states = current_hidden_states * top_k_weights[token_idx, 
                                                              top_k_pos, None]
final_hidden_states.index_add_(0, token_idx, 
                               current_hidden_states.to(final_hidden_states.dtype))
```

都处理完之后返回 final_hidden_states 即可

#### Qwen3_5MoeMLP

这个是 shared_expert 的 SwiGLU，更是精简

```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

---

### Qwen3_5MoeTextRotaryEmbedding

最后我们需要看一眼这里的 RoPE 实现

因为 qwen3.5 是多模态模型，因此使用的位置编码是 M-RoPE/Multimodal RoPE

原本的 RoPE 针对的是 1d 序列位置上的位置编码，M-RoPE 则编码三维位置：时间 T、高度 H、宽度 W。M-RoPE 的特色是将 head_dim 分成 THW 交错的部分，来使模型学习更容易。但在纯文本推理时，它的计算结果和标准 RoPE 完全一样

不过就算是标准 RoPE，在这里的计算方法也略有区别。假设一个 head 里每个 token 向量有 head_dim 维，我们总是取第 i 和第 i + head_dim / 2 维做旋转，而不是取 i 和 i + 1。下面简要推导一下这样做的结果

设

$$
\v x = (x_0, x_1,x_2,x_3,\dots)
$$

取

$$
\varphi_i = p\theta_i,\quad d=\text{head\_dim} / 2
$$

则

$$
\begin{align*}
f(\v x, p) &= \left(
\begin{bmatrix}\cos \varphi_0 & -\sin \varphi_0 \\ \sin \varphi_0 & \cos \varphi_0\end{bmatrix}
\begin{bmatrix} x_0 \\ x_d\end{bmatrix}, 
\begin{bmatrix}\cos \varphi_1 & -\sin \varphi_1 \\ \sin \varphi_1 & \cos \varphi_1\end{bmatrix}
\begin{bmatrix} x_1 \\ x_{1+d}\end{bmatrix},
\dots
\right) \\
\\
&= (x_0\cos\varphi_0-x_d\sin\varphi_0,\ x_1\cos\varphi_1-x_{1+d}\sin\varphi_1,\dots, \\
&\quad~~ x_0\sin\varphi_0+x_d\cos\varphi_0,\ x_1\sin\varphi_1+x_{1+d}\cos\varphi_1,\dots
)
\end{align*}
$$

上面是将两个维度旋转之后放回到原来的位置上之后的结果

可以注意到，整个变换可以写作这样：

$$
\begin{align*}
&\forall\ 0\le i\lt d, \\
x_i &\rightarrow x_i\cos\varphi_i - x_{i+d}\sin\varphi_i \\
x_{i+d} &\rightarrow x_{i+d}\cos\varphi_i + x_i\sin\varphi_i 
\end{align*}
$$

据此我们可以设计一种比较优雅的流程

1. 计算出 $\varphi_0,\varphi_1,\dots ,\varphi_{d-1}$ 一共 $d$ 个旋转角度

2. 复制一遍，变成 $\varphi_0,\varphi_1,\dots ,\varphi_{d-1},\varphi_0,\varphi_1,\dots ,\varphi_{d-1}$

3. 对上面的向量分别计算 cos 和 sin，得到

$$
\v c = (\cos\varphi_0,\cos\varphi_1,\dots,\cos\varphi_{d-1},~\cos \varphi_0,\cos\varphi_1,\dots,\cos\varphi_{d-1}) \\
\v s = (\sin\varphi_0,\sin\varphi_1,\dots,\sin\varphi_{d-1},~\sin \varphi_0,\sin\varphi_1,\dots,\sin\varphi_{d-1})
$$

4. 对原来的 $\v x$ 作 rotate_half 得到 $\v x'$

$$
\v x' = (-x_d,-x_{d+1},\dots,-x_{\text{head\_dim}-1},~x_0,x_1,\dots,x_{d-1})
$$

5. 最终结果为

$$
\text{res} = \v x\odot \v c + \v x'\odot \v s
$$

这个流程正是代码在做的事情

对于 text only 推理来说，Qwen3_5MoeTextRotaryEmbedding 计算的东西是这样的：

```python
freqs = inv_freq * position_ids  # 外积
emb = torch.cat([freqs, freqs], dim=-1)
cos = emb.cos()
sin = emb.sin()
```

最后返回 cos 和 sin。在 attention 中真正应用 RoPE 时就取出这个 cos 和 sin，对每个 token 计算

$$
\v x = \v x \odot \text{cos} + \text{rotate\_half}(x)\odot \text{sin}
$$

即可

---

### Cache

在搓的时候发现 cache 的实现需要观察一下

首先我们要知道 cache 里要存什么。对于普通 softmax attn 来说：

- 在 prefill 时把整个 K 和 V 都塞到 cache 里存起来
- 在 decode 时，假设所有 batch 的 seq_len == 1，我们只要把这个新 token 的 k、v 向量加入 cache，然后拿 q 去点乘现在及之前的所有 k，得到的权值加权现在及之前的所有 v 即可

对于 GDN 来说，forward 中涉及到跨多个 token 的操作为联想记忆矩阵 state 和对 mixed_qkv 要做的 conv1d 运算，因此我们可以策划如下的 cache 管理方案

- 在 prefill 时把 mixed_qkv 的最后 conv_kernel_size 个 token，以及最后得出的 state 塞到 cache 里
- 在 decode 时将新 token 的 mixed_qkv 向量追加到保存的 conv_cache 最后，同时把最老的 token 踢掉，这样维护出滑动窗口。对于 state 来说，基于旧的 state 更新一次再把新的 state 替换到 cache 里就可以了

观察 transformers 里的实现我们可以发现，cache 中存储的 tensor 也有 batch 维，因此这里我们隐式假设了<u>一个 completion 请求的 batch index 在推理全程都是固定不动的</u>