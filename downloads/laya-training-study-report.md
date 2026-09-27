---
title: Laya 训练方法学习报告
date: 2026-09-27
tags:
  - AI
  - Agent
  - 模型训练
  - 概率校准
aliases:
  - Laya训练方法
---

# Laya 训练方法学习报告

> [!summary] 一句话理解
> **Laya 把“生成答案”改造成“直接预测决策概率”：在预训练编码器上加入通用决策头，再用软标签监督学习、RLCD 和训练后温度缩放，让模型既能选对，也尽量诚实地表达不确定性。**

## 先记住六件事

1. Laya 不是从零学习语言，而是站在 ModernBERT / mmBERT 这类预训练编码器之上。
2. 它不生成文字，而是对本次提供的候选项直接打分，一次前向传播就能完成决策。
3. 候选项也是输入，所以它不是只能识别一套固定标签的普通分类器。
4. 训练标签可以是概率分布，而不只是 one-hot 标签。
5. 训练同时使用 **Soft Cross-Entropy** 和 **RLCD**：前者学习“答案在哪里”，后者学习“概率应该怎样分配”。
6. RLCD 追求校准不等于模型天然已经校准；真正上线前，仍要用自己的留出数据重新做温度缩放和阈值验证。

---

## 一、Laya 到底在解决什么问题？

大语言模型处理一个 Agent 决策时，通常要经历：

```text
理解上下文 → 推理 → 生成文字或 JSON → 解析输出 → 得到决策
```

但 Agent 运行过程中，很多步骤其实只是有限选项之间的判断：

- 工单应该交给哪个部门？
- 当前风险是低、中还是高？
- 下一步应该调用工具、追问用户，还是结束？
- 这个结果是否需要人工复核？

Laya 把问题直接定义成：

```text
P(decision | state, question, options)
```

于是流程缩短为：

```text
理解上下文 → 输出决策概率
```

它的目标不是替代会推理、会生成的大模型，而是接管 Agent 中大量高频、重复、选项明确的“快判断”。可以把它理解成 Agent 的 **System 1 判断脑**。

---

## 二、它不是从零训练 4 亿参数

英文 Laya 使用 ModernBERT-large 作为编码器：

```text
ModernBERT-large 编码器   约 395M 参数
+ Laya 决策模块
= Laya                    约 421M 参数
```

ModernBERT 已经通过大规模预训练获得了语言理解能力。官方资料显示，它先在约 1.7T Token、1024 长度下训练，再经过 250B Token 的 8192 长上下文适配和 50B Token 的退火阶段。

因此，Laya 所做的不是重新教模型理解“退款”“诈骗”“带宽”等词，而是把已经会读语言的编码器训练成一个专业判断员。

需要区分两个训练层次：

| 层次 | 起点 | 做了什么 | 公开程度 |
| --- | --- | --- | --- |
| Laya base 的形成 | ModernBERT / mmBERT | 加入并训练通用决策架构 | 完整原始训练流程和数据未全部公开 |
| `laya-typed-decisions` 专项微调 | 已训练好的 `convaiinnovations/laya` | 用领域数据继续全量微调 | 公开 notebook 可复现 |

也就是说，公开 notebook **不是从原始 ModernBERT 开始训练 Laya**，而是从已有 Laya base checkpoint 继续做领域专门化。

---

## 三、Laya 怎样把不同任务变成同一种“考试”？

### 1. 普通分类器的问题

传统分类器通常是：

```text
文本 → 编码器 → 固定维度分类头 → 固定类别
```

如果原来只有 `billing / technical / sales`，后来新增 `security / legal / refund`，输出层往往也要跟着修改和重新训练。

### 2. Laya 把候选项也写进输入

Laya 的实际序列格式是：

```text
[CLS]
<问题类型> question: <问题说明>
[SEP]
[MASK] 选项 0
[MASK] 选项 1
...
[SEP]
state
[SEP]
```

例如：

```text
State:
“我的信用卡被重复扣款了。”

Question:
应该由哪个部门处理？

Options:
[MASK] billing: 支付和账单问题
[MASK] technical: 软件问题
[MASK] sales: 购买咨询
[MASK] security: 安全事件
```

模型读取每个 `[MASK]` 位置的向量，为每个候选项产生一个 logit，再经过 softmax 得到概率。

更准确地说，结构不是简单的“BERT 后面接一个 MLP”，而是：

```text
预训练双向编码器
      ↓
加入 choice / score / noul 类型嵌入
      ↓
2 层 Transformer 决策头
      ↓
提取每个 [MASK] marker 的隐藏向量
      ↓
MLP scorer：每个选项 → 1 个 logit
      ↓
softmax → 决策概率分布
```

因为选项是输入而不是固定输出神经元，同一套模型可以处理：

- `choice`：多个选项中选择一个；
- `score`：在有顺序的等级上评分；
- `noul`：判断 yes / no 的概率。

这就是它被称为“通用分类器”的原因。但要注意：**输出形式通用，不代表业务能力天然通用。** 新领域仍然通常需要专门数据微调。

---

## 四、训练样本不是“文本 → 标签”，而是一道完整决策题

一个训练样本更接近：

```text
State
+ Question
+ Options / Criteria
+ Gold probability distribution
```

例如：

```text
State:
Customer says: "I've been charged twice."

Question:
Which team should handle this?

Options:
billing / technical / sales

Target:
billing    0.95
technical  0.03
sales      0.02
```

这里最值得学习的是：目标可以是软概率，而不必是 one-hot。

```text
传统硬标签：A = 1，B = 0，C = 0
Laya 软标签：A = 0.70，B = 0.20，C = 0.10
```

软标签保留了教师对歧义、次优方案和不确定性的判断。公开 notebook 会读取每个选项的 `probabilities`，并把它归一化为和为 1 的目标分布。

公开的 typed-decisions 微调数据包括：

- 1,200 个训练 case；
- 每个 case 包含多道 typed decision；
- 合计 6,000 个决策训练项；
- 覆盖 Agent trace、客户服务、发票处理和安全事件四类合成工作流。

### 这些软概率究竟从哪里来？

`typed-decisions` 数据集采用的是“合成场景 + 教师模型多次采样”的方法：

```text
1. 随机采样隐变量骨架
   例如主题、语气、严重度、客户年限、是否违反约束
                    ↓
2. 根据骨架生成具体 state
   文本场景由模型写出，发票和 Agent trace 等保留结构化形式
                    ↓
3. 把同一个 case 交给教师端点回答 3 次
   sampling temperature = 0.7
                    ↓
4. 每次都要求返回完整概率分布
                    ↓
5. 对三次概率逐项取平均，作为 gold soft distribution
```

例如，同一道三选一问题，教师模型三次给出：

```text
第 1 次：A 0.80，B 0.15，C 0.05
第 2 次：A 0.60，B 0.30，C 0.10
第 3 次：A 0.75，B 0.15，C 0.10
```

最终训练标签是：

```text
A = (0.80 + 0.60 + 0.75) / 3 ≈ 0.717
B = (0.15 + 0.30 + 0.15) / 3 = 0.200
C = (0.05 + 0.10 + 0.10) / 3 ≈ 0.083
```

这里平均的是每次返回的**完整概率分布**，不是只统计三次 argmax。假如只统计第一名，上例会变成 `A=1，B=0，C=0`，教师对 B、C 的犹豫就全部丢失了。

数据集还保存了 `label_agreement`，用于记录三次教师采样之间的分歧。温度 0.7 则让教师产生一定变化：太低会让三次答案几乎一样，太高又会引入过多随机性。

> [!warning] 软概率不是客观真相
> 这里的 gold 是约 4B 级教师端点的三次平均，因此模型训练出来的是“逼近教师的概率判断”。它不是人群真实选择频率，也不是业务结果的客观发生概率。数据集官方明确说明：在该 benchmark 上得分，衡量的是与教师的一致性，不等于现实正确性；教师甚至漏掉过一个 ID 完全相同的重复发票。

因此，如果我们自己构造训练数据，优先级应该是：

1. **真实业务结果的经验分布**：最接近真实概率，但需要足够历史样本；
2. **多名领域专家独立标注**：用投票比例或经校准的专家置信度形成分布；
3. **多个强教师、多次采样后聚合**：兼顾成本与规模，但要抽样人工审查；
4. **单个教师单次输出概率**：成本最低，也最容易把教师的偏见和虚假置信度写入学生模型；
5. **硬标签加固定 label smoothing**：只能防止极端自信，不能表达每个样本真实不同的歧义。

---

## 五、完整训练流程：先让方向正确，再让概率诚实

```text
已有 Laya base checkpoint
          ↓
构造 State + Question + Options + 概率标签
          ↓
编码器与决策头前向传播，得到 logits
          ↓
┌──────────────────┬──────────────────┐
│ Soft Cross-Entropy│       RLCD       │
│ 学习目标分布在哪里 │ 学习怎样报告概率   │
└──────────────────┴──────────────────┘
          ↓
联合反向传播，全量微调
          ↓
在未参与训练的数据上拟合 Temperature
          ↓
评测准确率、Brier、ECE、MAE 与业务覆盖率
```

下面重点拆解 RLCD。

---

## 六、RLCD：在“概率分布空间”里做强化学习

RLCD 的全称是：

> Reinforcement Learning for Calibrated Decisions

它想解决的问题是：模型即使选对了，也可能过度自信。

目标分布如果是：

```text
A 70%　B 20%　C 10%
```

两组预测都把 A 排第一：

```text
预测 1：A 99%　B 0.5%　C 0.5%
预测 2：A 72%　B 18% 　C 10%
```

普通准确率认为两者一样；Laya 希望奖励预测 2，因为它更忠实地表达了任务的不确定性。

### 第一步：给 logits 加探索噪声

公开 notebook 对当前 logits 加零均值高斯噪声，一次生成 4 组候选分布：

```text
GROUP_SIZE = 4
σ：从 0.4 逐步降到 0.1
```

直觉上：训练初期多探索，后期逐渐稳定。

这里探索的不是不同文本回答，而是不同的**概率分布**。这也是它比生成式 RL 轻量得多的原因。

### 第二步：用严格适当评分规则给每组概率打分

Laya 的 reward 由三部分组成：

$$
R(q,y)=\underbrace{\sum_i y_i\log q_i}_{\text{Log Score}}
+\alpha\underbrace{\frac{y\cdot q}{\lVert q\rVert}}_{\text{Spherical Score}}
-\mathbb{1}_{score}\beta\underbrace{RPS(q,y)}_{\text{有序等级距离}}
$$

其中：

- $y$：目标概率分布；
- $q$：模型报告的概率分布；
- Log Score 惩罚把真实可能性压得过低；
- Spherical Score 奖励预测分布与目标方向一致；
- RPS 只用于有顺序的 `score` 问题。

公开 typed-decisions notebook 实际使用：

```text
w_sph = 0.75
w_rps = 1.0
```

#### 1. Log Score：最怕“自信地排除正确可能性”

$$
R_{log}=\sum_i y_i\log q_i
$$

它其实就是 Soft Cross-Entropy 的相反数。因为 $\log q_i\le 0$，reward 越接近 0 越好。

它有一个非常重要的特性：当目标在某个选项上仍有概率，而模型把该选项压到接近 0 时，惩罚会迅速变大。

```text
目标认为 B 仍有 20% 可能

模型报 B = 18%：可以接受
模型报 B = 0.5%：受到很大惩罚
模型报 B = 0：理论上 log(0) → -∞
```

源码为数值稳定做了两层保护：先把概率截到至少 `1e-12`，再把 log reward 的下限截为 `-9.21`，约等于 `log(0.0001)`。这避免一个极小概率让 loss 爆炸；严格来说，也会让极端尾部的“严格适当性”稍有弱化。

#### 2. Spherical Score：看整条分布的方向是否一致

$$
R_{sph}=\frac{y\cdot q}{\lVert q\rVert}
$$

分子是目标分布与预测分布的点积，分母对预测向量长度做归一化。直觉上，它在问：

> 预测概率向量是否朝向目标概率向量？

它比 Log Score 平滑、范围也更稳定，可以补充 Log Score 对极小概率非常敏感的特点。由于 $y$ 对当前样本固定，且 $q$ 的概率和为 1，这一项在 $q$ 与 $y$ 同方向时达到最好；最终就是鼓励 $q$ 接近 $y$。

#### 3. RPS：有序等级中不仅看“错没错”，还看“错多远”

对于 `score` 问题，源码额外计算：

$$
RPS=\frac{1}{K-1}\sum_k\left(CDF_q(k)-CDF_y(k)\right)^2
$$

然后从 reward 中减掉它：

$$
R=R_{log}+0.75R_{sph}-RPS
$$

假设真实等级为 4：

```text
预测主要落在等级 3：RPS ≈ 0.141
预测主要落在等级 0：RPS ≈ 0.796
```

虽然两者的 argmax 都错了，但前者只差一级，应该比后者少罚。普通无序 Cross-Entropy 不知道等级 3 比等级 0 更接近等级 4，RPS 知道。

#### 4. 一个完整数值例子

目标分布：

```text
y = [0.70, 0.20, 0.10]
```

按照源码中的 `Log + 0.75 × Spherical` 计算：

| 预测分布 q | Log Score | Spherical | 总 Reward | 解读 |
| --- | ---: | ---: | ---: | --- |
| `[0.99, 0.005, 0.005]` | -1.5965 | 0.7015 | **-1.0704** | 第一名虽对，但严重过度自信 |
| `[0.72, 0.18, 0.10]` | -0.8032 | 0.7344 | **-0.2523** | 最接近目标，得分最高 |
| `[0.60, 0.25, 0.15]` | -0.8245 | 0.7270 | **-0.2793** | 稍保守，但仍比较合理 |
| `[0.40, 0.40, 0.20]` | -0.9856 | 0.6333 | **-0.5106** | 判断不够明确，得分下降 |

这里 reward 都是负数完全没有问题。训练只关心谁更大：`-0.2523` 高于 `-1.0704`，所以第二组明显优于第一组。

RPS 的价值在于表达“错多远”：真实风险是 4 时，预测 3 应该比预测 0 少受惩罚。

严格适当评分规则的核心性质是：

> 从长期期望得分看，最优策略是诚实报告自己相信的概率，而不是故意把概率报得更极端。

### 第三步：组内比较，计算相对优势

如果 4 组候选 reward 是：

```text
0.3，1.2，0.8，-0.1
```

模型会减去组内平均值，并用标准差归一化：

```text
Advantage = Reward - Group Mean
```

高于组内平均的分布被鼓励，低于平均的分布被抑制。这是 GRPO-style / group-mean baseline 的含义。

源码中还会把 Advantage 除以标准差，使不同 batch 的 reward 尺度更稳定。

接下来，RLCD 把 noisy logits 看成从以当前 logits 为均值的高斯策略中采样：

$$
\log \pi(z\mid l)\propto-\frac{\lVert z-l\rVert^2}{2\sigma^2}
$$

Policy Gradient loss 是：

$$
L_{RLCD}=-\mathbb{E}\left[A(z)\log\pi(z\mid l)\right]
$$

- $A(z)>0$：这次噪声方向比组内平均好，推动基础 logits 靠近它；
- $A(z)<0$：这次方向更差，推动基础 logits 远离它；
- 减去组均值不会改变期望梯度，但能显著降低方差。

噪声还会减去各选项上的均值，成为 zero-mean projection。这是因为给所有 logits 同时加同一个常数不会改变 softmax；去掉这个无效的公共方向，可以把探索集中到真正改变概率分布的方向上。

但它与大语言模型的 GRPO 有本质区别：

| 生成式 GRPO | Laya RLCD |
| --- | --- |
| 探索不同文本序列 | 探索不同概率分布 |
| 动作空间是 Token | 动作空间是 noisy logits |
| 奖励完整回答 | 奖励概率报告 |
| 计算成本高 | 单次编码后可轻量采样 |

### 第四步：RLCD 与监督损失联合训练

Laya 并不是纯强化学习。公开 notebook 的损失是：

$$
L=L_{RLCD}+1.0\times L_{SoftCE}
$$

其中：

$$
L_{SoftCE}=-\sum_i y_i\log p_i
$$

最容易记忆的解释是：

```text
Soft CE：告诉模型“正确分布在哪里”
RLCD：  告诉模型“怎样探索并奖励更诚实的概率”
```

二者不是互相替代，而是共同作用。

> [!note] 一个值得保持清醒的判断
> 当前 reward 本身对概率是可微的，也可以直接反向传播；Laya 选择用 noisy logits + REINFORCE，是在优化“噪声探索后的期望 reward”，并为未来接入不可微业务奖励保留统一形式。因此，RLCD 不是凭空创造了新的标签信息：它使用的仍是同一个教师目标分布，只是换了一种带探索和相对基线的优化方式。真正决定上限的，仍然是目标概率的质量。

---

## 七、为什么要全量微调？为什么使用两组学习率？

公开 typed-decisions notebook 会同时更新编码器和决策头，而不是只训练最后一层。

```text
Encoder LR      = 2.5e-5
Decision Head LR = 1.0e-4
```

决策头的学习率是编码器的 4 倍：

- 编码器已经掌握语言能力，只需小步适应领域；
- 决策头更贴近新任务，需要更快调整。

公开训练配置如下：

| 项目 | 配置 |
| --- | ---: |
| Epoch | 4 |
| GPU | Kaggle T4 × 2，DDP |
| 每卡 micro batch | 8 |
| 梯度累积 | 4 |
| 有效 batch | 64 个决策序列 |
| RLCD group size | 4 |
| 噪声 σ | 0.4 → 0.1 |
| Encoder LR | 2.5e-5 |
| Head LR | 1.0e-4 |
| Optimizer | AdamW，weight decay 0.01 |
| Scheduler | Cosine Annealing |
| 梯度裁剪 | 1.0 |
| 精度 | FP16 autocast |

Notebook 还对编码器和决策头都启用了 gradient checkpointing，用更多计算换取更低显存占用。

---

## 八、训练结束后为什么还要做 Temperature Scaling？

训练正确并不意味着置信度可靠。

如果模型报出 0.90 置信度的样本，长期只有 70% 正确，那么 0.90 就没有业务意义。Temperature Scaling 的做法是：

$$
p=softmax(z/T)
$$

- $T>1$：让分布变平，降低过度自信；
- $T<1$：让分布变尖，提高置信度；
- 一般不改变 argmax，因此主要校准概率，不主要改变分类答案。

公开 notebook 从训练项中预先留出最多 400 条、约 10% 的决策项，训练结束后分别为 `choice / score / noul` 拟合温度。

> [!warning] 最重要的落地提醒
> RLCD 的训练目标是校准概率，但这不等于公开 checkpoint 的概率可以直接拿来设自动化阈值。官方模型卡明确提醒 `laya-typed-decisions` 仍然过度自信，ECE 为 0.213；使用前应在自己的 held-out data 上重新拟合温度，并测量“置信度阈值—准确率—覆盖率”的关系。

例如不能直接规定：

```text
confidence > 0.85 就自动执行
```

正确做法是先在自有数据上回答：

```text
阈值 0.85 时，实际准确率是多少？
能自动处理多少比例的请求？
错误成本能否接受？
数据漂移后是否仍成立？
```

---

## 九、一个电路设计 Agent 的训练例子

### State

```text
当前 CTLE 仿真：
Gain = 8.1 dB
BW = 9.3 GHz
Eye height = 41 mV

目标：
Gain > 10 dB
BW > 8 GHz
Eye height > 60 mV

上一轮：增大 R1 后 BW 下降
```

### Question

```text
下一步应该做什么？
```

### Options 与教师分布

| 动作 | Target |
| --- | ---: |
| A 增大 gm | 0.55 |
| B 增大 load R | 0.05 |
| C 调整 peaking inductor | 0.30 |
| D 重新选择 topology | 0.08 |
| E 结束优化 | 0.02 |

模型要学习的不是单独一个 `A`，而是完整概率结构。这样 Agent 可以建立分层决策：

```text
Laya 输出概率
     ↓
高置信度、低风险 → 自动执行
中等置信度       → 便宜推理模型复核
低置信度或高风险 → 强推理模型 / 人工介入
```

真正的价值不是“把大模型全部替掉”，而是把不值得每次调用大模型的判断筛出来。

---

## 十、从 Laya 中最值得学习的五个训练思想

### 1. 先改变问题定义，再考虑扩大模型

如果任务本质是有限决策，就直接优化决策概率，不必绕道生成自然语言。

### 2. 把标签语义放进输入

这让模型能面对动态选项，也使标签描述成为可以设计、审查和版本管理的任务接口。

### 3. 保留不确定性，而不是强迫所有样本 one-hot

软标签能表达歧义、多解和教师不一致。但前提是教师概率本身有质量，不能只把单次 LLM 输出伪装成“真实概率”。

### 4. 将“准确”与“可信”分开优化、分开评测

- Accuracy：第一名选对没有？
- Brier / Log score：完整概率分布好不好？
- ECE：报出的置信度是否与实际正确率一致？
- Coverage：给定安全阈值后，可以自动处理多少任务？

### 5. 专家小模型的价值来自领域数据闭环

公开 benchmark 中，base Laya 的准确率约为 0.362；领域微调后的 `laya-typed-decisions` 达到约 0.766。它更像“可培养的专业学生”，而不是拿来即用的通才。

---

## 十一、不要误解的地方

1. **“通用分类器”不等于零样本能力很强。** 通用的是输入输出形式，业务能力仍依赖微调。
2. **RLCD 不保证自动校准成功。** 奖励设计、数据质量、分布漂移和温度拟合都会影响最终概率。
3. **公开 benchmark 主要是四类合成工作流。** 不能直接外推到电路设计、企业流程或其他真实业务。
4. **模型报告的 benchmark 不是独立第三方结论。** 特别是与 Jev 的比较，官方明确说明样本和 prompt 不完全一致，只能作为参考。
5. **低延迟不等于端到端系统一定快。** 模型加载、Tokenize、批处理、网络服务和路由同样影响实际延迟。
6. **概率阈值不是模型常量。** 它是基于业务错误成本、校准数据和期望覆盖率制定的策略。

---

## 十二、一页复习卡

### 训练链路

```text
预训练 Encoder
→ State + Question + Options
→ [MASK] marker 表示每个候选项
→ Transformer decision head + MLP scorer
→ logits / probability distribution
→ Soft CE 学方向
→ RLCD 学概率报告
→ 全量微调
→ held-out temperature scaling
→ 用 Accuracy + Brier + ECE + Coverage 验证
```

### 三个关键公式

```text
Soft CE： -Σ yᵢ log pᵢ

RLCD Reward：Log Score + Spherical Score - RPS（仅有序评分）

Calibration：softmax(logits / T)
```

### 三个核心区别

| 对比对象 | Laya 的区别 |
| --- | --- |
| 普通固定分类器 | 选项也是输入，可以处理动态候选项 |
| 生成式 LLM | 不生成 Token，直接输出决策概率 |
| 纯监督分类 | Soft CE 之外，再在概率分布空间做 RLCD |

### 每次复习时问自己

1. 为什么把 option 写进输入后，输出层可以不绑定固定标签？
2. Soft CE 和 RLCD 各自解决什么问题？
3. 为什么 proper scoring rule 会鼓励诚实报告概率？
4. Laya 的 GRPO-style 与生成式 GRPO 有什么根本区别？
5. 为什么训练后还必须做温度缩放？
6. 为什么 ECE 好不等于 accuracy 高，accuracy 高也不等于可以自动执行？
7. 如果用于电路设计 Agent，我应该怎样获得高质量软标签和留出校准集？

---

## 我的最终理解

Laya 最值得关注的不是 421M 参数，也不只是两层 Transformer 决策头，而是它把 Agent 的一类问题重新定义了：

> **不是让模型“说出应该做什么”，而是让模型直接估计“每个动作有多值得做”。**

这带来三个结果：

- 去掉语言生成，速度和成本更适合高频调用；
- 保留完整概率分布，为路由、升级和人工介入提供依据；
- 用领域数据训练后，可以成为嵌入具体工作流的专用判断脑。

真正可迁移到我们工作中的训练方法是：**先把复杂任务拆成结构清晰的决策题，再通过软标签、概率奖励、校准和业务阈值，把“模型会判断”变成“系统敢使用”。**

## 资料来源

- [Laya GitHub 仓库](https://github.com/NandhaKishorM/laya)
- [公开 typed-decisions 微调 notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
- [Laya 核心模型、序列构造与 proper reward 源码](https://github.com/NandhaKishorM/laya/blob/main/laya/common.py)
- [Laya Typed-Decisions 模型卡与 benchmark](https://huggingface.co/convaiinnovations/laya-typed-decisions)
- [ModernBERT 官方介绍与预训练流程](https://huggingface.co/blog/modernbert)

> 核对时间：2026-09-27。仓库和模型卡会持续更新，具体参数应以对应版本源码为准。
