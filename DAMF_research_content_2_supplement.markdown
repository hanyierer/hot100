下面详细介绍这篇论文 **Query Rewriting for Retrieval-Augmented Large Language Models**。它的核心思想很直接：**RAG 系统不一定只优化检索器或大模型，也可以先把用户问题改写成更适合检索的查询，再进行检索和回答**。论文提出的框架叫 **Rewrite-Retrieve-Read**，即“改写—检索—阅读”。

## 1. 论文要解决什么问题？

传统 RAG 通常是：

```text
用户问题 Query → Retriever 检索文档 → Reader/LLM 根据文档回答
```

论文称这种流程为 **Retrieve-then-Read**。

问题在于：**用户原始问题不一定适合作为检索查询**。例如用户问：

> What profession does Nicholas Ray and Elia Kazan have in common?

如果直接把这句话送进搜索引擎，检索结果可能不稳定。更好的检索查询可能是：

```text
Nicholas Ray profession; Elia Kazan profession
```

这样检索器更容易找到两个人的职业信息，LLM 再根据检索结果回答 “director”。

所以论文认为，RAG 中有三个可优化对象：

```text
Query → Retriever → Reader
```

已有很多工作优化 Retriever 或 Reader，但这篇论文重点优化 **Query 本身**。

------

## 2. 核心贡献

论文的贡献主要有两个：

第一，提出 **Rewrite-Retrieve-Read** 框架，在检索之前加入 Query Rewriting 模块。

第二，除了用 ChatGPT 直接改写 Query，论文还训练了一个较小的可学习模型作为 **Query Rewriter**，让它适配固定的搜索引擎和固定的黑盒大模型。

整体流程是：

```text
原始问题 x
   ↓
Query Rewriter 生成改写查询 x~
   ↓
搜索引擎检索相关文档 doc
   ↓
ChatGPT 读取 [doc, x]
   ↓
生成最终答案 ŷ
```

这里 Retriever 和 Reader 都是冻结的，真正训练的是前面的 Rewriter。

------

## 3. 方法整体结构

论文对比了三种主要流程。

### 3.1 标准 Retrieve-then-Read

这是普通 RAG：

```text
Question → Web Search → Documents → Black-box LLM → Answer
```

用户问题直接送入搜索引擎，然后把检索结果拼接到 prompt 中，让 ChatGPT 回答。

这个方法简单，但缺点是：**问题表达不一定适合搜索引擎**。

------

### 3.2 用 LLM 作为 Frozen Rewriter

第二种方法是让 ChatGPT 先改写查询：

```text
Question → ChatGPT 生成 Search Query → Web Search → Documents → ChatGPT → Answer
```

例如 prompt 会要求：

```text
Think step by step to answer this question, and provide search engine queries...
```

这种方式不训练模型，只依赖 ChatGPT 的 few-shot 能力。论文实验发现，仅仅加入这一步，很多任务效果就提升了。

但它也有问题：

1. ChatGPT 生成的 Query 可能不稳定；
2. 改写结果依赖 prompt；
3. 调用大模型做改写成本较高；
4. 无法针对某个固定 Retriever 和 Reader 进行专门优化。

因此作者进一步提出可训练的 Rewriter。

------

### 3.3 可训练 Query Rewriter

第三种方法是论文的重点：

```text
Question → Trainable Rewriter → Web Search → Documents → ChatGPT → Answer
```

这个 Rewriter 是一个小语言模型。英文任务中使用 **T5-large，770M 参数**；中文任务中使用 **mT5-large，1.2B 参数**。

注意：这里说“小模型”是相对于 ChatGPT 这类黑盒大模型而言，不是说它真的很小。T5-large 仍然是一个比较大的 Seq2Seq 模型。

------

## 4. 训练流程

论文的训练分为两步：

```text
第一步：监督学习 warm-up
第二步：强化学习 PPO 微调
```

------

## 4.1 第一步：监督学习 Warm-up

Query Rewriting 不是普通语言模型预训练任务，所以作者先用伪数据进行监督训练。

伪数据构造流程如下：

```text
原始问题 x
   ↓
用 ChatGPT 生成改写查询 x~
   ↓
用 x~ 搜索文档
   ↓
ChatGPT 根据文档回答
   ↓
如果回答正确，则保留 (x, x~)
```

也就是说，只有当某个改写 Query 最终帮助系统答对问题时，这条改写数据才会被保留下来，作为训练样本。

得到伪训练集：

```text
D~ = {(x, x~) | ŷ = y}
```

然后用标准 Seq2Seq 交叉熵训练 Rewriter，让模型学习从原始问题生成改写查询。

论文中的监督学习损失是：

```text
L_ll = - Σ_t log pθ(x~_t | x~_<t, x)
```

含义是：给定原始问题 x，让模型逐 token 预测正确的改写查询 x~。

这一步可以看作是让 Rewriter 先学会“像 ChatGPT 那样改写查询”。

------

## 4.2 第二步：强化学习 PPO 微调

监督学习只能模仿 ChatGPT 的改写，但不一定最适合当前 RAG 系统。因此作者把整个 RAG 流程看成一个强化学习环境。

在这个环境里：

| 强化学习概念 | 在论文中的含义                  |
| ------------ | ------------------------------- |
| Policy       | Query Rewriter                  |
| State        | 当前已经生成的 Query token      |
| Action       | 生成下一个 token                |
| Episode      | 生成完整 Query 后结束           |
| Reward       | 最终 RAG 回答效果               |
| Environment  | Retriever + Reader 整个黑盒流程 |

具体来说，Rewriter 生成一个查询，搜索引擎检索文档，ChatGPT 回答问题，然后根据回答质量给 Rewriter 奖励。

如果最终答案正确，奖励高；如果答案错误，奖励低。

------

## 4.3 Reward 设计

对于开放域问答任务，论文使用：

```text
R_lm = EM + λ_f F1 + λ_h h
```

其中：

- **EM**：Exact Match，答案是否完全匹配；
- **F1**：预测答案与标准答案的词级重合度；
- **h**：检索文档是否包含正确答案，如果包含则为正，否则为负。

这说明作者不仅关注最终回答是否正确，也关注检索结果是否真的命中了答案。

对于多选题任务，奖励为：

```text
R_lm = EM + λ_f F1
```

因为多选题主要看选项是否预测正确。

------

## 4.4 为什么要加 KL 正则？

论文还加入了 KL-divergence 正则项：

```text
R = R_lm - β KL(πθ || π0)
```

其中：

- πθ 是当前 Rewriter；
- π0 是 warm-up 后的初始 Rewriter；
- KL 项限制模型不要偏离初始模型太远。

原因是强化学习容易把语言模型训练坏，例如生成奇怪的、不自然的 Query。KL 正则可以让模型在优化任务奖励的同时，仍然保持基本语言能力和合理输出格式。

------

## 5. Retriever 和 Reader 怎么设计？

### 5.1 Retriever：Bing 搜索引擎

论文没有使用 Milvus、FAISS 或 DPR 这类本地向量检索器，而是直接使用 **Bing Search API**。

作者认为这样有两个好处：

1. 不需要自己维护索引；
2. 能获取实时网页知识。

检索结果使用两种方式处理：

第一种是直接使用搜索结果页的 snippet，也就是搜索引擎返回的摘要。

第二种是进入网页抓取正文，然后用 BM25 按查询相关性筛选文本，保留更相关的内容。

实验表明，**BM25 过滤后的网页正文比 snippet 效果更好**。

------

### 5.2 Reader：ChatGPT

Reader 使用的是 **gpt-3.5-turbo**。

Prompt 基本形式为：

```text
Instruction + Demonstrations + Input
```

对于 RAG，Input 是：

```text
[retrieved documents, original question]
```

也就是说，虽然检索用的是改写后的 Query，但最后让 ChatGPT 回答时，仍然会提供原始问题 x，避免模型只根据改写 Query 回答而丢失用户原意。

这一点很重要：**改写 Query 是为了检索，不是为了替换用户问题**。

------

## 6. 实验任务

论文在两类任务上评估。

### 6.1 开放域问答

包括：

1. **HotPotQA**：多跳问答，需要组合多个事实；
2. **AmbigNQ**：存在歧义的开放域问题；
3. **PopQA**：包含较多长尾知识的问题。

这些任务都很适合测试 Query Rewriting，因为原始问题往往不够直接，或者需要补充背景知识。

------

### 6.2 多选题任务

包括：

1. **gaokao**：中国高考题，选择历史、地理、生物、物理等更依赖知识的科目；
2. **JEC-QA**：中国司法考试问答；
3. **MMLU**：英文多任务考试题。

多选题中，系统需要根据问题和选项生成适合搜索的查询，然后根据检索结果选择答案。

------

## 7. 实验结果

论文主要比较四种方法：

| 方法                  | 含义                                     |
| --------------------- | ---------------------------------------- |
| Reader                | 不检索，直接让 ChatGPT 回答              |
| Retrieve-then-read    | 原始问题直接检索，然后回答               |
| LLM rewriter          | ChatGPT 先改写 Query，再检索回答         |
| Rewrite-retrieve-read | 训练后的 Rewriter 改写 Query，再检索回答 |

------

## 7.1 HotPotQA 结果

在 HotPotQA 上：

| 方法                  | EM    | F1    |
| --------------------- | ----- | ----- |
| Reader                | 32.36 | 43.05 |
| Retrieve-then-read    | 30.47 | 41.34 |
| LLM rewriter          | 32.80 | 43.85 |
| Rewrite-retrieve-read | 33.70 | 44.87 |

可以看到一个很有意思的现象：普通检索增强反而比不检索更差。

这说明检索并不一定总是提升效果。如果检索结果不相关，或者给 LLM 引入噪声，反而会干扰回答。

但加入 Query Rewriting 后，性能超过 Reader baseline；训练后的 Rewriter 效果最好。

------

## 7.2 AmbigNQ 和 PopQA 结果

在 AmbigNQ 上：

| 方法               | EM    | F1    |
| ------------------ | ----- | ----- |
| Reader             | 42.10 | 53.05 |
| Retrieve-then-read | 45.80 | 58.50 |
| LLM rewriter       | 46.50 | 58.95 |

在 PopQA 上：

| 方法               | EM    | F1    |
| ------------------ | ----- | ----- |
| Reader             | 41.94 | 44.61 |
| Retrieve-then-read | 43.20 | 47.53 |
| LLM rewriter       | 46.00 | 49.74 |

这里说明：对于歧义问题和长尾知识问题，检索本身就有帮助，而 Query Rewriting 能进一步提升检索质量。

------

## 7.3 多选题结果

在中文多选任务上：

| 方法               | gaokao | JEC  |
| ------------------ | ------ | ---- |
| Reader             | 48.3   | 21.1 |
| Retrieve-then-read | 55.81  | 24.4 |
| LLM rewriter       | 58.18  | 28.4 |

在 MMLU 上：

| 方法               | Humanities | STEM  | Other |
| ------------------ | ---------- | ----- | ----- |
| Reader             | 75.60      | 58.80 | 69.00 |
| Retrieve-then-read | 76.70      | 63.30 | 70.00 |
| LLM rewriter       | 77.00      | 63.50 | 72.60 |

这些结果说明 Query Rewriting 不只适用于开放域问答，也适用于考试类、多选类知识任务。

------

## 8. 论文的关键发现

### 8.1 Query Rewriting 的作用不是“润色问题”

它的核心不是把问题改得更自然，而是把问题改得更适合检索。

例如原问题可能是：

```text
What profession does Nicholas Ray and Elia Kazan have in common?
```

更好的检索 Query 是：

```text
Nicholas Ray profession; Elia Kazan profession
```

这不是语言润色，而是把问题拆成搜索引擎更容易理解的关键词组合。

------

### 8.2 RAG 的瓶颈不一定在大模型

很多人做 RAG 时会优先考虑：

```text
换更强的 LLM
换更好的 embedding
换更好的 reranker
加更复杂的 prompt
```

但这篇论文提醒我们：**检索 Query 的质量本身就非常关键**。

如果 Query 不好，后面的 Retriever 再强也可能召回不到正确内容。

------

### 8.3 直接检索可能引入噪声

HotPotQA 上，Retrieve-then-read 低于 Reader baseline，这说明检索增强不是无脑提升。

RAG 有一个常见问题：

```text
错误文档 + 强制让 LLM 参考文档 = 更差答案
```

Query Rewriting 可以缓解这个问题，因为它能让检索结果更聚焦。

------

### 8.4 可训练 Rewriter 比单纯 prompt 更有扩展性

用 ChatGPT 改写 Query 简单有效，但每次都要调用大模型，成本和延迟较高。

训练一个专门的 Rewriter 后，可以在本地或较低成本环境中运行：

```text
用户 Query → 小模型改写 → 检索 → 大模型回答
```

这对工程系统更友好。

------

## 9. 论文的不足

这篇论文也有一些局限。

第一，Retriever 使用 Bing 搜索，实验更偏向 Web RAG，对企业内部知识库、Milvus、Elasticsearch、向量数据库等场景没有深入验证。

第二，训练 Rewriter 的强化学习成本较高。每次生成 Query 后，都要经过搜索引擎和 ChatGPT 才能得到 reward，这在真实系统里会产生较高 API 成本。

第三，实验中可训练 Rewriter 的完整结果主要体现在 HotPotQA 上，其他任务更多展示的是 LLM rewriter 的提升，说明 RL 训练部分的扩展验证还不够充分。

第四，论文主要关注 Query Rewriting，对 rerank、chunk 策略、混合检索、RRF 融合等现代 RAG 关键模块讨论较少。

------

## 10. 对你做 RAG 系统的启发

这篇论文对实际 RAG 系统很有参考价值，尤其是你之前做的“意图识别、Query 改写/扩展、Milvus 检索、重排序、LLM 生成”流程。

可以借鉴成下面这个结构：

```text
用户问题
   ↓
意图识别
   ↓
Query Rewriting / Query Expansion
   ↓
多路检索：
   - 原始 Query 检索
   - 改写 Query 检索
   - 关键词 Query 检索
   - HyDE Query 检索
   ↓
RRF 融合去重
   ↓
Reranker 精排
   ↓
LLM 生成答案
```

工程上不一定一开始就用 PPO 训练 Rewriter，可以分阶段做：

### 第一阶段：规则 + LLM 改写

让大模型根据不同意图生成改写 Query，例如：

```text
资料类：抽取核心实体 + 主题词
资讯类：加入时间、最新、公告、版本等词
攻略类：加入步骤、打法、推荐、技巧等词
```

### 第二阶段：监督微调 Rewriter

收集线上日志：

```text
用户原始问题
改写 Query
检索结果
用户是否点击
答案是否被采纳
```

构造训练数据，微调 T5、BART、Qwen 小模型等。

### 第三阶段：奖励优化

等系统有足够数据后，再考虑类似论文的强化学习思路，用最终 RAG 效果反向优化 Query Rewriter。

------

## 11. 一句话总结

这篇论文的核心价值在于：**它把 RAG 优化的关注点从“检索器和大模型”前移到了“用户问题本身”，证明了 Query Rewriting 可以显著改善检索质量，并进一步提升最终问答效果。**

对于实际 RAG 系统来说，它最值得借鉴的不是一定要用 PPO，而是这个思想：

```text
不要直接相信用户原始 Query 适合检索；
应该先把 Query 改写成更适合知识库或搜索引擎的形式。
```





你这次上传的压缩包是一个 **RAG Query Rewriting 项目代码**，目录名是：

```text
RAG-query-rewriting-main
```

它对应前面那篇论文 **Query Rewriting for Retrieval-Augmented Large Language Models** 的实现思路：在传统 RAG 的“检索—阅读”之前，增加一个 **Query Rewriter 查询改写模块**，让原始问题先变成更适合搜索引擎或知识库检索的查询，再进行检索和大模型回答。论文中称这个框架为 **Rewrite-Retrieve-Read**。

------

# 1. 这个项目整体在做什么？

项目目标是实现下面这套流程：

```text
原始问题 Question
   ↓
Query Rewriting 查询改写
   ↓
Bing / Wikipedia 检索
   ↓
检索结果整理 / BM25 筛选
   ↓
LLM 读取检索结果并回答
   ↓
计算 EM / F1 等指标
```

如果只看核心实验，它主要支持四类模式：

```text
1. Reader only
   不检索，直接让 LLM 回答问题

2. Retrieve-then-read
   原始问题直接检索，再让 LLM 根据检索结果回答

3. LLM as Rewriter
   用 LLM 先改写查询，再检索，再回答

4. Trainable Rewriter
   用 T5 / mT5 等可训练模型生成改写查询，再检索，再回答
```

这和论文中的方法是一致的：**不是只优化检索器，也不是只优化大模型，而是优化检索前的 Query 表达**。

------

# 2. 项目目录结构

压缩包里主要目录如下：

```text
RAG-query-rewriting-main/
├── README.md
├── overview.png
├── examples/
│   ├── run_eg.sh
│   └── epoch_20_test_split_predictions.json
├── generate/
│   ├── main-0420.py
│   ├── main-0514.py
│   ├── main-llama.py
│   ├── inference.py
│   ├── interllama.py
│   ├── evaluation.py
│   ├── bingenv.py
│   ├── wikienv.py
│   ├── bing.py
│   ├── bing_utils/
│   │   ├── bing.py
│   │   └── bm25skl.py
│   └── inprompts/
│       ├── myprompt.jsonl
│       └── regular.jsonl
├── rl/
│   ├── RLcode.md
│   ├── run.sh
│   └── RL4LMs/
│       ├── rl4lms/
│       └── scripts/
└── result_models/
    └── README.md
```

可以把它理解成三大部分：

```text
generate/       负责普通推理、检索、Query 改写、评估
rl/             负责强化学习训练 Query Rewriter
examples/       给出运行脚本和预测结果示例
result_models/ 说明训练好的模型 checkpoint 地址
```

------

# 3. 应该从哪里开始看？

建议按这个顺序看：

## 第一步：README.md

README 介绍了项目目标：实现 **Rewrite-Retrieve-Read**，用查询改写提升检索增强大模型的效果。

它还说明项目借鉴了：

```text
GenRead
ReAct
RL4LMs
Vicuna
Bing API
OpenAI API
```

其中最重要的是 **RL4LMs**，因为论文中的可训练 Rewriter 是通过强化学习 PPO 训练的。

------

## 第二步：examples/run_eg.sh

这个文件给出了核心运行命令。

里面分成三类：

```bash
# direct
python main-0420.py --dataset hotpot --task step1 ...

# LLM rewrite
python main-0420.py --dataset hotpot --task rewrite ...
python main-0420.py --dataset hotpot --task rewrite2 ...

# trainable rewrite
python scripts/training/train_text_generation.py --config_path ...
```

含义是：

| 命令类别          | 作用                                       |
| ----------------- | ------------------------------------------ |
| direct            | 不进行 Query Rewriting，直接问答或直接检索 |
| LLM rewrite       | 用 LLM 生成改写 Query，然后检索回答        |
| trainable rewrite | 用 PPO 训练 T5/mT5 作为可训练 Rewriter     |

------

## 第三步：generate/main-0420.py 和 generate/main-0514.py

这两个是普通推理实验的主入口。

其中：

```text
main-0420.py
```

主要用于 HotPotQA、TriviaQA、NQ、AmbigNQ、PopQA 等开放域问答任务。

```text
main-0514.py
```

在 main-0420.py 基础上增加了更多任务支持，比如 MMLU、JEC-QA、gaokao 等多选题任务。

------

# 4. generate 目录详细介绍

## 4.1 main-0420.py / main-0514.py：实验主入口

这两个文件负责把完整流程串起来。

里面最重要的函数有：

| 函数              | 作用                                       |
| ----------------- | ------------------------------------------ |
| `datapath()`      | 根据 dataset 和 split 找到数据集文件路径   |
| `readfiles()`     | 读取 json / jsonl 数据                     |
| `step1()`         | Reader-only，不检索，直接让 LLM 回答       |
| `wiki()`          | 原始问题检索，再让 LLM 回答                |
| `rewrite()`       | 先让 LLM 生成改写 Query，再检索            |
| `rewrite2()`      | 读取 rewrite 阶段的检索结果，再让 LLM 回答 |
| `searchrewrite()` | 先检索，再让 LLM 对检索结果做摘要，再回答  |
| `bing_bl()`       | Bing 检索 baseline                         |

其中最核心的是 `rewrite()` 和 `rewrite2()`。

------

## 4.2 `rewrite()`：生成改写 Query 并检索

`rewrite()` 做的是第一阶段：

```text
原始问题
   ↓
LLM 生成改写 Query
   ↓
用改写 Query 进行 Bing / Wiki 检索
   ↓
保存检索结果
```

例如原始问题：

```text
What profession does Nicholas Ray and Elia Kazan have in common?
```

可能被改写成：

```text
Nicholas Ray profession; Elia Kazan profession
```

然后系统会分别搜索这些 Query，并把检索结果保存成 jsonl 文件。

这个阶段还支持 `--think` 参数。如果开启 `--think`，模型输出会包含 Thought 和 Query，例如：

```text
Thought: I need to search Nicholas Ray and Elia Kazan, find their professions.
Query: Nicholas Ray profession; Elia Kazan profession
```

代码会从输出中解析 `Query:` 后面的内容。

------

## 4.3 `rewrite2()`：根据改写检索结果回答

`rewrite2()` 是第二阶段：

```text
读取 rewrite 阶段保存的检索结果
   ↓
把检索结果作为 background/context
   ↓
让 LLM 回答原始问题
   ↓
计算 EM / F1
```

注意：最终回答时仍然使用原始问题，而不是只用改写后的 Query。

这点很重要。Query Rewriting 的作用是 **帮助检索**，不是替代用户原始问题。

------

## 4.4 inference.py：调用 LLM 的推理模块

`generate/inference.py` 主要负责：

```text
构造 Prompt
批量调用 LLM
把输出写入 jsonl
```

核心函数：

| 函数                | 作用                                                |
| ------------------- | --------------------------------------------------- |
| `add_prompt()`      | 把 `{query}`、`{background}` 等占位符替换成真实内容 |
| `complete()`        | 调用 LLM API 的函数                                 |
| `run_main()`        | 对一批样本进行 LLM 推理                             |
| `run_searchre()`    | 对检索结果进行摘要或重写                            |
| `run_main_search()` | 检索增强问答推理                                    |

但是这个文件里有一个关键问题：

```python
def complete(...):
    pass
```

也就是说，**当前压缩包里的 OpenAI / ChatGPT 调用函数没有真正实现**。

如果你想跑通这个项目，需要自己补上 `complete()`，例如改成调用 OpenAI API、豆包 API、通义 API、Qwen API，或者本地 vLLM 服务。

------

## 4.5 interllama.py：本地 LLaMA 推理

`generate/interllama.py` 是本地 LLaMA 模型版本。

它使用：

```python
LlamaForCausalLM
LlamaTokenizer
```

主要流程是：

```text
加载本地 LLaMA 模型
   ↓
构造 prompt
   ↓
model.generate()
   ↓
截取 Answer 后面的输出
   ↓
保存结果
```

对应入口是：

```text
generate/main-llama.py
```

也就是说，如果不想用 OpenAI API，可以尝试走 `main-llama.py + interllama.py` 这套本地推理流程。

不过这份代码使用的是较旧的 LLaMA 接口，实际运行时可能需要根据你本地 transformers 版本做适配。

------

## 4.6 bingenv.py：Bing 检索环境

`generate/bingenv.py` 封装了一个类似环境的 Bing 检索器。

它支持这些 action：

```text
search[xxx]
lookup[xxx]
finish[xxx]
think[xxx]
```

最重要的是：

```python
step(action, use_en, func, gold, topn=None, max_words_perdoc=None, black=None)
```

其中 `func` 可以控制检索方式：

| func          | 含义                               |
| ------------- | ---------------------------------- |
| `plain`       | 直接用 Bing 搜索结果 snippet       |
| `bm25`        | 进入网页正文后用 BM25 选相关段落   |
| `gold`        | 用答案命中方式筛选文档，偏实验分析 |
| `r1`          | 只取第一个搜索结果                 |
| `plainfilter` | 用于 MMLU，支持过滤污染 URL        |

它的作用是把搜索引擎封装成一个可以被主程序调用的 Retriever。

------

## 4.7 bing_utils/bing.py：更完整的 Bing 检索工具

这个文件比 `generate/bing.py` 更完整。

它主要包括：

| 函数           | 作用                             |
| -------------- | -------------------------------- |
| `searchbing()` | 调用 Bing Search API             |
| `morer()`      | 根据 Bing 结果进入网页，抓取正文 |
| `searchsele()` | 搜索后用 BM25 选择相关段落       |
| `searchbl()`   | 根据 gold answer 选择命中文档    |
| `searchrdoc()` | 返回拼接后的网页正文             |
| `searchr1()`   | 返回第一个网页结果               |

其中 `searchsele()` 是论文中“网页正文 + BM25 筛选”思路的实现。

流程是：

```text
Bing 搜索
   ↓
获取搜索结果 URL
   ↓
requests 访问网页
   ↓
BeautifulSoup 提取 <p> 段落
   ↓
BM25 按 Query 选择最相关段落
   ↓
返回给 LLM 作为上下文
```

------

## 4.8 bing_utils/bm25skl.py：BM25 实现

这个文件用 `sklearn.TfidfVectorizer` 实现了 BM25。

核心类：

```python
class BM25
```

核心函数：

```python
bm25score(docs, q, max_words, topp, use)
```

作用是从多个候选段落中选出和查询最相关的文本。

它支持两种选择方式：

| use     | 含义                                           |
| ------- | ---------------------------------------------- |
| `words` | 按 BM25 分数从高到低累加段落，直到超过最大词数 |
| `topp`  | 选取接近最高分的一批段落                       |

这部分在 RAG 中对应：

```text
粗召回结果 → 文本筛选 / 压缩 → LLM 上下文
```

------

## 4.9 wikienv.py：Wikipedia 检索环境

`wikienv.py` 和 `bingenv.py` 类似，不过它面向 Wikipedia。

它支持：

```text
search[entity]
lookup[keyword]
finish[answer]
```

这部分比较像 ReAct 里的 Wikipedia 环境。

区别是：

```text
wikienv.py 适合固定 Wikipedia 检索
bingenv.py 适合真实网页搜索
```

------

## 4.10 evaluation.py：评估模块

这个文件负责计算问答效果。

主要指标：

| 指标         | 含义                                        |
| ------------ | ------------------------------------------- |
| EM           | Exact Match，预测答案和标准答案是否完全匹配 |
| F1           | 预测答案和标准答案的词级重合度              |
| Recall / hit | 检索结果是否包含答案                        |
| Rouge-L      | 主要用于生成类任务                          |

重要函数：

```python
eval_question_answering()
eval_recall()
eval_fact_checking()
eval_dialogue_system()
```

对于开放域问答，主要看：

```text
EM + F1
```

这也和论文实验保持一致。

------

# 5. prompts 目录详细介绍

`generate/inprompts/` 里有两个 prompt 文件：

```text
regular.jsonl
myprompt.jsonl
```

其中 `myprompt.jsonl` 更重要，里面定义了不同任务的 prompt 模板。

例如 Reader-only prompt：

```text
Answer the question in the following format, end the answer with '**'.

Question: {query}

Answer:
```

普通检索增强 prompt：

```text
Question: {background} {query}

Answer:
```

Query Rewriting prompt：

```text
Provide a better search query for web search engine to answer the given question, end the query with '**'.

Question: {query}

Query:
```

还有带思维过程的 Query Rewriting prompt：

```text
Think step by step and rewrite better search queries for a web search engine to answer the given question.
Split the queries with ';' and end the query with '**'.

Question: {query}
Thought:
```

这说明项目支持两种 Query 改写方式：

```text
1. 直接生成 Query
2. 先生成 Thought，再生成 Query
```

论文里的例子也是第二种：先分析需要搜索什么，再生成搜索 Query。

------

# 6. rl 目录详细介绍

`rl/` 目录是训练可学习 Rewriter 的部分。

它基于 **RL4LMs**，核心目标是：

```text
用 PPO 训练 T5 / mT5，让它生成更适合检索的 Query
```

整体训练流程是：

```text
原始问题
   ↓
T5 Rewriter 生成 Query
   ↓
Bing 检索
   ↓
LLM 回答
   ↓
计算 EM / F1 / hit
   ↓
把结果作为 reward
   ↓
PPO 更新 Rewriter
```

这正是论文中的强化学习训练思想。

------

## 6.1 RL 训练入口

训练入口在：

```text
rl/RL4LMs/scripts/training/train_text_generation.py
```

这个文件读取 YAML 配置，然后根据算法类型创建 trainer：

```python
if "supervised" in config["alg"]["id"]:
    trainer = SupervisedTrainer(...)
else:
    trainer = OnPolicyTrainer(...)
```

如果是 supervised，就是监督微调。

如果是 ppo / nlpo / trpo，就是强化学习训练。

------

## 6.2 PPO 配置文件

核心配置在：

```text
rl/RL4LMs/scripts/training/task_configs/hotpot/
```

例如：

```text
t5_ppo_0314_v2.yml
t5_ambig_0525_v3.yml
t5_pop_0609_v3.yml
mt5_jec_v2.yml
other_0610.yml
```

以 `t5_ppo_0314_v2.yml` 为例，里面配置了：

```yaml
tokenizer:
  model_name: /xinbei_data/replug/baseline/experiments/t5l-hotpot/

reward_fn:
  id: llmv3

datapool:
  id: hotpot
  args:
    prompt_prefix: "rewrite a better search query: "

alg:
  id: ppo
  args:
    n_steps: 512
    batch_size: 16
    learning_rate: 0.000002
    n_epochs: 2
```

可以看出它训练的是一个 **T5 查询改写模型**，输入前缀是：

```text
rewrite a better search query:
```

也就是模型输入类似：

```text
rewrite a better search query: What profession does Nicholas Ray and Elia Kazan have in common?
```

输出是：

```text
Nicholas Ray profession; Elia Kazan profession
```

------

## 6.3 DataPool：训练数据加载

训练数据类在：

```text
rl/RL4LMs/rl4lms/data_pools/custom_text_generation_pools.py
```

里面定义了很多数据集类：

```text
TQA
hotpot
jec
Ambig
popqa
mmlusocial
mmluhumanities
mmluother
mmlustem
```

每个类都继承自：

```python
TextGenPool
```

它们负责把原始数据转成统一格式：

```python
Sample(
    id=...,
    prompt_or_input_text=...,
    references=...
)
```

也就是说，RL 训练时不直接读原始 json，而是先转成 RL4LMs 需要的 `Sample` 格式。

------

# 7. examples 目录介绍

## 7.1 run_eg.sh

这个文件是命令示例。

它包含三类实验：

### Reader-only / Retrieve-then-read

```bash
python main-0420.py --dataset hotpot --task step1 ...
python main-0420.py --dataset hotpot --task wiki ...
```

### LLM Query Rewriting

```bash
python main-0420.py --dataset hotpot --task rewrite ...
python main-0420.py --dataset hotpot --task rewrite2 ...
```

### Trainable Rewriter

```bash
python scripts/training/train_text_generation.py \
  --config_path scripts/training/task_configs/hotpot/t5_ppo_0314_v2.yml \
  --experiment_name xx
```

------

## 7.2 epoch_20_test_split_predictions.json

这个文件是训练后的 Query Rewriter 预测示例。

里面每条样本大概长这样：

```json
{
  "prompt_text": "rewrite a better search query: Were Scott Derrickson and Ed Wood of the same nationality?",
  "generated_text": "Scott Derrickson nationality; Ed Wood nationality",
  "ref_text": "<START-1> ['yes'] <END-1>",
  "em": true,
  "f1": 1.0
}
```

这里的含义是：

```text
输入：原始问题
输出：模型生成的改写 Query
评价：把这个 Query 用于检索和回答后，最终答案是否正确
```

注意，`em` 和 `f1` 评价的是最终问答结果，不是 Query 字面相似度。

这点非常重要：**Query Rewriter 的目标不是生成“看起来像标准答案”的 Query，而是生成能让 RAG 最终答对的 Query。**

------

# 8. result_models 目录介绍

`result_models/README.md` 里面给了模型 checkpoint 地址，包括：

```text
Trainable rewriter after warmup
Trainable rewriter after RL
```

这对应论文中的两个阶段：

```text
1. Warm-up supervised training
2. PPO reinforcement learning
```

也就是说，作者不是从零开始用 PPO 训练，而是先用伪数据监督微调，再用最终 RAG 效果作为 reward 做 PPO。

------

# 9. 这个项目的完整运行流程

如果按论文实验走，整体应该是下面这样。

## 9.1 Baseline 1：不检索，直接回答

```text
Question
   ↓
LLM
   ↓
Answer
```

对应命令：

```bash
python main-0420.py \
  --dataset hotpot \
  --task step1 \
  --split dev \
  --promptfile generate/inprompts/myprompt.jsonl \
  --pid 3 \
  --output_dir outputs
```

作用：测试纯 LLM 的能力。

------

## 9.2 Baseline 2：原始问题直接检索

```text
Question
   ↓
Bing Search
   ↓
Documents
   ↓
LLM
   ↓
Answer
```

对应命令：

```bash
python main-0420.py \
  --dataset hotpot \
  --task wiki \
  --split dev \
  --promptfile generate/inprompts/myprompt.jsonl \
  --pid 3 \
  --search bing \
  --output_dir outputs
```

作用：测试普通 RAG。

------

## 9.3 LLM Query Rewriting 第一阶段

```text
Question
   ↓
LLM 生成 Query
   ↓
Bing Search
   ↓
保存检索结果
```

对应命令：

```bash
python main-0420.py \
  --dataset hotpot \
  --task rewrite \
  --split dev \
  --promptfile generate/inprompts/myprompt.jsonl \
  --pid 6 \
  --search bing \
  --output_dir outputs \
  --think
```

这里 `--think` 表示使用带思考过程的 Query Rewriting。

------

## 9.4 LLM Query Rewriting 第二阶段

```text
读取改写 Query 的检索结果
   ↓
LLM 根据检索结果回答
   ↓
计算 EM / F1
```

对应命令：

```bash
python main-0420.py \
  --dataset hotpot \
  --task rewrite2 \
  --split dev \
  --promptfile generate/inprompts/myprompt.jsonl \
  --pid 1 \
  --search bing \
  --repid 6 \
  --output_dir outputs
```

其中 `--repid` 对应前面 rewrite 阶段使用的 prompt id。

------

## 9.5 Trainable Rewriter 训练

```text
T5 / mT5 Rewriter
   ↓
生成 Query
   ↓
Bing 检索
   ↓
LLM 回答
   ↓
EM/F1/hit 作为 reward
   ↓
PPO 更新 Rewriter
```

对应命令示例：

```bash
cd rl/RL4LMs

python scripts/training/train_text_generation.py \
  --config_path scripts/training/task_configs/hotpot/t5_ppo_0314_v2.yml \
  --experiment_name t5l-hotpot-ppo
```

------

# 10. 这个项目目前不能直接运行的地方

这份代码更像论文实验代码，不是一个开箱即用的工程项目。主要问题如下。

## 10.1 数据路径是硬编码的

例如：

```python
/xinbei_data/replug/ReAct/data/hotpot_dev_v1_simplified.json
/xinbei_data/replug/ambignq/dev.jsonl
/xinbei_data/replug/generate/popqa/test.jsonl
```

这些路径是作者服务器上的路径。

你本地运行时必须改成自己的数据路径，例如：

```python
data/hotpot/hotpot_dev_v1_simplified.json
data/ambignq/dev.jsonl
```

建议把 `datapath()` 改成从命令行传入：

```bash
--input_file data/hotpot_dev.json
```

而不是写死在代码里。

------

## 10.2 OpenAI 调用函数没有实现

`generate/inference.py` 中：

```python
def complete(...):
    pass
```

这意味着如果你直接跑 `main-0420.py`，到调用 LLM 时不会真正生成答案。

你需要自己实现：

```text
OpenAI API
豆包 API
通义 API
Qwen API
本地 vLLM API
```

任选一种。

------

## 10.3 Bing API 需要配置环境变量

`generate/bing_utils/bing.py` 使用：

```python
os.environ['BING_SEARCH_V7_SUBSCRIPTION_KEY']
os.environ['BING_SEARCH_V7_ENDPOINT']
```

所以需要配置：

```bash
export BING_SEARCH_V7_SUBSCRIPTION_KEY="你的 Bing Key"
export BING_SEARCH_V7_ENDPOINT="https://api.bing.microsoft.com/"
```

Windows PowerShell 下是：

```powershell
$env:BING_SEARCH_V7_SUBSCRIPTION_KEY="你的 Bing Key"
$env:BING_SEARCH_V7_ENDPOINT="https://api.bing.microsoft.com/"
```

------

## 10.4 RL4LMs 代码可能不完整

`train_text_generation.py` 里导入了：

```python
from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import OnPolicyTrainer, SupervisedTrainer
```

但是当前压缩包里没有看到完整的：

```text
rl4lms/envs/text_generation/
```

这说明当前包里的 RL4LMs 可能不是完整版本，或者需要额外安装原始 RL4LMs。

所以强化学习部分如果直接跑，可能会报：

```text
ModuleNotFoundError: No module named 'rl4lms.envs'
```

解决方式是：安装完整 RL4LMs，或者把缺失的 `envs/text_generation` 模块补齐。

------

# 11. 如果你要把它改成自己的 RAG 系统，应该怎么改？

你之前做的是游戏知识库 RAG，使用 Milvus、BGE-M3、reranker、LLM。这个项目可以借鉴思想，但不建议直接照搬 Bing Web Search。

可以改成下面这样：

```text
用户 Query
   ↓
Query Rewriter
   ↓
Milvus 向量检索
   ↓
BM25 / 稀疏检索
   ↓
RRF 融合
   ↓
bge-reranker 精排
   ↓
LLM 生成答案
```

对应替换关系：

| 原项目模块        | 你的系统中替换为                         |
| ----------------- | ---------------------------------------- |
| Bing Search       | Milvus dense vector search               |
| Bing snippet      | 知识库 chunk                             |
| BM25 网页段落筛选 | 稀疏检索 / BM25 / SPLADE / BGE-M3 sparse |
| ChatGPT Reader    | 豆包 / Qwen / DeepSeek / 本地 vLLM       |
| T5 Rewriter       | T5 / Qwen / BART / BERT意图识别+模板改写 |
| EM/F1 reward      | 用户点击、答案采纳、LLM judge、人工标注  |

------

# 12. 对你最有用的部分

如果你是为了实现 RAG 中的 Query 改写，最值得借鉴的是这几块：

## 12.1 Prompt 改写模板

看：

```text
generate/inprompts/myprompt.jsonl
```

尤其是 `task = rewrite` 的 prompt。

你可以改成游戏 RAG 场景：

```text
请将用户问题改写为适合游戏知识库检索的查询。
要求：
1. 保留英雄名、技能名、装备名、版本号等关键实体；
2. 删除口语化表达；
3. 如果是攻略类问题，补充“打法、出装、铭文、连招、克制”等检索词；
4. 如果是资讯类问题，补充“版本、公告、更新、调整”等检索词；
5. 输出 3 个检索 Query，用分号分隔。
```

------

## 12.2 多 Query 检索思想

论文和代码都支持：

```text
Query1; Query2; Query3
```

然后分别检索，再合并上下文。

这对 RAG 很实用。比如用户问：

```text
澜现在怎么玩，怎么出装？
```

可以改写成：

```text
澜 打法技巧;
澜 出装推荐;
澜 铭文 连招;
澜 当前版本 强度
```

然后多路检索，再 RRF 融合。

------

## 12.3 用最终问答效果评价 Query

这个项目最重要的思想是：

```text
Query 改写好不好，不看 Query 本身像不像标准答案；
而看它能不能让最终 RAG 答对。
```

这对工程系统非常重要。

你可以记录：

```text
原始 Query
改写 Query
召回 chunk
rerank 分数
最终答案
用户是否满意
```

然后把这些日志反过来训练 Query Rewriter。

------

# 13. 总结

这个压缩包实现的是一个 **RAG 查询改写实验框架**，核心是：

```text
不要直接拿用户原始问题去检索；
先把问题改写成更适合检索器的 Query；
再检索、再让 LLM 回答。
```

项目中最核心的代码是：

```text
generate/main-0420.py       开放域问答主流程
generate/main-0514.py       多任务 / 多选题扩展流程
generate/inference.py       LLM 推理接口
generate/bingenv.py         Bing 检索环境
generate/bing_utils/bing.py Bing 搜索 + 网页抓取 + BM25筛选
generate/evaluation.py      EM / F1 / Recall 评估
generate/inprompts/         Prompt 模板
rl/RL4LMs/                  PPO 训练可学习 Query Rewriter
```

但它不是开箱即用项目，主要需要补：

```text
1. 修改硬编码数据路径
2. 实现 complete() 大模型调用函数
3. 配置 Bing API
4. 补齐或安装完整 RL4LMs
5. 根据你的 RAG 系统替换 Bing 为 Milvus / ES / 混合检索
```

对于你自己的 RAG 系统，最推荐吸收的是它的思想和流程：

```text
原始 Query
   ↓
意图识别
   ↓
Query 改写 / 扩展
   ↓
多路检索
   ↓
融合去重
   ↓
rerank
   ↓
LLM 生成
```

其中 **Query Rewriting** 可以先用 prompt 实现，后续再用日志数据微调 T5/Qwen 小模型，最后再考虑用奖励优化。





可以，而且**非常适合**。你完全可以把这份代码里“微调 T5 做 Query Rewriting”的思想用于**游戏知识库 RAG 系统的 Query 改写**，并且**不使用强化学习部分**。

论文和代码本身其实也是两阶段思路：第一阶段是用伪数据对 Rewriter 做监督微调 warm-up，第二阶段才是 PPO 强化学习。你现在只保留第一阶段即可，也就是：

```text
用户原始问题 → 微调后的 T5 Rewriter → 改写 Query → Milvus / BM25 / 混合检索 → rerank → LLM 生成
```

论文中也明确提到，Rewriter 先通过伪数据进行监督 warm-up，然后才进一步用强化学习优化；所以工程落地时完全可以只使用监督微调部分。

------

# 1. 能不能用于游戏知识库 RAG？

答案是：**能，而且比原论文场景更适合工程落地**。

原论文面向开放域 QA 和 Web Search，例如：

```text
What profession does Nicholas Ray and Elia Kazan have in common?
```

改写成：

```text
Nicholas Ray profession; Elia Kazan profession
```

你的游戏知识库 RAG 中，用户问题通常更加口语化，例如：

```text
澜现在怎么玩？
```

如果直接拿这句话去 Milvus 检索，可能召回不稳定。更好的改写结果应该是：

```text
澜 打法技巧; 澜 出装推荐; 澜 铭文搭配; 澜 连招技巧; 澜 当前版本强度
```

这样检索器可以从多个角度召回知识库内容。

因此，T5 Rewriter 在你的系统里可以承担这个角色：

```text
把口语化、模糊、不完整的用户问题
改写成更适合知识库检索的结构化查询
```

------

# 2. 不用强化学习是否可行？

完全可行。

你可以只做：

```text
监督微调 T5 / mT5 / 中文 T5
```

不做：

```text
PPO
reward_fn
actor-critic
KL penalty
RL4LMs 强化学习训练
```

也就是说，原代码中的强化学习目录可以先不用：

```text
rl/RL4LMs/algorithms/ppo/
rl/RL4LMs/algorithms/a2c/
rl/RL4LMs/algorithms/trpo/
reward_fn
env
OnPolicyTrainer
```

只保留思想：

```text
输入：原始 Query
输出：改写后的检索 Query
训练方式：Seq2Seq 监督微调
```

------

# 3. 在游戏 RAG 中，T5 Rewriter 应该输出什么？

建议不要只输出一个 Query，而是输出**多个检索 Query**。

例如用户问：

```text
孙尚香打马可波罗怎么打？
```

T5 改写为：

```text
孙尚香 对线 马可波罗 技巧; 孙尚香 克制 马可波罗; 孙尚香 打射手 对线思路; 马可波罗 弱点
```

这样后续可以多路检索：

```text
Query 1 → Milvus dense 检索
Query 2 → Milvus dense 检索
Query 3 → 稀疏检索 / BM25
Query 4 → 稀疏检索 / BGE-M3 sparse
```

然后做：

```text
RRF 融合 → 去重 → reranker 精排 → LLM 生成
```

------

# 4. 推荐的整体系统流程

你原来的游戏知识库 RAG 可以改成下面这样：

```text
用户问题
   ↓
Query 预处理
   ↓
意图识别
   ↓
T5 Query Rewriter
   ↓
生成多个改写 Query
   ↓
多路检索
   ├── 原始 Query dense 检索
   ├── 改写 Query dense 检索
   ├── 改写 Query sparse 检索
   └── 关键词检索 / BM25
   ↓
RRF 融合去重
   ↓
bge-reranker 精排
   ↓
LLM 生成答案
```

其中 T5 Rewriter 放在 **意图识别之后、检索之前**。

------

# 5. Query Rewriter 的输入输出设计

## 5.1 输入格式

建议输入给 T5 的文本加上任务前缀，例如：

```text
rewrite game rag query: 澜现在怎么玩？
```

或者更细一点，把意图也拼进去：

```text
rewrite game rag query intent=攻略类: 澜现在怎么玩？
```

如果你的系统已经有意图识别模块，可以把意图结果作为 T5 的额外输入。

例如：

```text
rewrite game rag query intent=资料类: 孙权技能是什么？
rewrite game rag query intent=资讯类: 最近王者荣耀更新了什么？
rewrite game rag query intent=攻略类: 澜现在怎么玩？
```

这样模型会学到不同意图下的改写方式。

------

## 5.2 输出格式

推荐输出多个 Query，用分号分隔：

```text
澜 打法技巧; 澜 出装推荐; 澜 铭文搭配; 澜 连招技巧; 澜 当前版本强度
```

不要一开始就让模型输出复杂 JSON，因为 T5 输出 JSON 容易格式不稳定。

更稳妥的方式是：

```text
query1; query2; query3; query4
```

然后在 Java 或 Python 里按 `;` 切分。

------

# 6. 不同意图下的改写策略

你的游戏 RAG 里常见意图可以分为：

```text
资料类
资讯类
攻略类
```

可以让 T5 学习不同的改写模式。

## 6.1 资料类 Query 改写

用户问题：

```text
孙策的一技能是什么？
```

目标改写：

```text
孙策 一技能 技能介绍; 孙策 劈风斩浪 技能效果; 孙策 技能机制
```

适合检索：

```text
英雄技能介绍
技能数值
技能机制
被动 / 一技能 / 二技能 / 三技能
```

------

## 6.2 资讯类 Query 改写

用户问题：

```text
最近李白加强了吗？
```

目标改写：

```text
李白 版本更新 加强; 李白 调整公告; 李白 技能数值改动; 李白 当前版本改动
```

适合检索：

```text
版本公告
英雄调整
装备调整
赛季更新
平衡性改动
```

------

## 6.3 攻略类 Query 改写

用户问题：

```text
澜怎么玩？
```

目标改写：

```text
澜 打法技巧; 澜 出装推荐; 澜 铭文搭配; 澜 连招技巧; 澜 打野思路
```

适合检索：

```text
英雄打法
出装铭文
连招技巧
克制关系
对线思路
团战思路
```

------

# 7. 训练数据怎么构造？

不使用强化学习时，最关键的是构造监督训练数据：

```text
原始 Query → 改写 Query
```

建议使用 JSONL 格式。

## 7.1 数据格式示例

```json
{"input": "rewrite game rag query intent=攻略类: 澜现在怎么玩？", "target": "澜 打法技巧; 澜 出装推荐; 澜 铭文搭配; 澜 连招技巧; 澜 打野思路"}
{"input": "rewrite game rag query intent=资料类: 孙策一技能是什么？", "target": "孙策 一技能 技能介绍; 孙策 劈风斩浪 技能效果; 孙策 技能机制"}
{"input": "rewrite game rag query intent=资讯类: 最近李白加强了吗？", "target": "李白 版本更新 加强; 李白 调整公告; 李白 技能数值改动; 李白 当前版本改动"}
{"input": "rewrite game rag query intent=攻略类: 鲁班七号怕什么英雄？", "target": "鲁班七号 克制英雄; 鲁班七号 被谁克制; 鲁班七号 对线弱点; 鲁班七号 生存技巧"}
{"input": "rewrite game rag query intent=资料类: 破军有什么效果？", "target": "破军 装备效果; 破军 属性介绍; 破军 被动效果; 破军 适合英雄"}
```

------

# 8. 训练数据来源

你可以用三种方式构造数据。

## 8.1 人工标注少量高质量样本

先人工写 300～1000 条。

覆盖：

```text
英雄技能
装备效果
铭文搭配
出装推荐
连招技巧
克制关系
版本更新
皮肤活动
地图机制
段位机制
```

这部分数据质量最高，用来做验证集也很合适。

------

## 8.2 用大模型批量生成伪数据

可以让大模型根据你的知识库标题、chunk 内容、常见用户问题自动生成：

```text
原始口语问题
标准改写 Query
```

例如给大模型 prompt：

```text
你是游戏知识库 RAG 系统的数据构造助手。
请根据下面的知识片段，生成 5 个用户可能提出的口语问题，并为每个问题生成适合知识库检索的改写 Query。
要求改写 Query 用分号分隔，保留英雄名、技能名、装备名、版本号等关键词。
```

知识片段：

```text
英雄：澜
内容：澜适合打野，核心连招为 12A3A...
```

生成：

```json
{
  "input": "rewrite game rag query intent=攻略类: 澜怎么连招？",
  "target": "澜 连招技巧; 澜 技能释放顺序; 澜 打野 连招; 澜 12A3A 连招"
}
```

------

## 8.3 从线上日志构造数据

系统上线后记录：

```text
用户原始问题
用户点击的文档
最终答案是否被采纳
检索命中的 chunk
```

然后反推出好的改写 Query。

例如用户问：

```text
猴子怎么秒人？
```

用户最终点击的是：

```text
孙悟空 爆发连招 技巧
```

那么可以构造训练样本：

```json
{
  "input": "rewrite game rag query intent=攻略类: 猴子怎么秒人？",
  "target": "孙悟空 爆发连招; 孙悟空 秒人技巧; 孙悟空 出装铭文; 孙悟空 连招顺序"
}
```

------

# 9. 推荐训练集规模

可以按阶段来做。

| 阶段       | 数据量         | 目标                           |
| ---------- | -------------- | ------------------------------ |
| Demo 阶段  | 300～1000 条   | 验证 T5 Rewriter 是否有效      |
| 初版可用   | 3000～10000 条 | 覆盖主要英雄、装备、攻略、资讯 |
| 较稳定版本 | 2万～5万条     | 覆盖长尾问题和不同表达方式     |
| 线上增强   | 10万条以上     | 用真实日志持续优化             |

如果你只是做毕业设计、课程项目或软件著作权，**3000～10000 条已经足够展示完整方案**。

------

# 10. 模型选择

如果游戏知识库主要是中文，建议不要直接用英文 T5-large。

可以选择：

| 模型                | 适用情况                                       |
| ------------------- | ---------------------------------------------- |
| mT5-small           | 资源有限，快速实验                             |
| mT5-base            | 推荐入门版本，效果和速度均衡                   |
| mT5-large           | 效果更强，但显存和推理成本较高                 |
| 中文 T5 / Mengzi-T5 | 中文任务更合适                                 |
| Qwen 小模型 SFT     | 如果你想统一成大模型风格，也可以用 Qwen 做改写 |

如果你只是做 Query 改写，不需要太大的模型。推荐优先级：

```text
mT5-base / 中文 T5-base > mT5-small > T5-large
```

原因是：

```text
T5-large 英文能力强，但中文游戏 Query 改写不一定最合适；
mT5 / 中文 T5 对中文输入输出更自然；
Query 改写任务较短，不需要特别大的模型。
```

------

# 11. 监督微调训练方式

不使用 RL 时，可以直接用 HuggingFace `Seq2SeqTrainer` 微调。

训练目标就是：

```text
输入原始 Query
生成改写 Query
```

损失函数就是标准交叉熵：

```text
L = -Σ log p(target_token | input, previous_target_tokens)
```

这对应论文里的 warm-up 思路，即让 Rewriter 学会从原始问题生成改写查询。

------

# 12. 简化版训练代码思路

下面是核心训练逻辑示例。

```python
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

model_name = "google/mt5-base"  # 可替换为中文T5或本地模型路径

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/game_query_rewrite_train.jsonl",
        "validation": "data/game_query_rewrite_val.jsonl"
    }
)

max_input_len = 128
max_target_len = 128

def preprocess(batch):
    inputs = batch["input"]
    targets = batch["target"]

    model_inputs = tokenizer(
        inputs,
        max_length=max_input_len,
        truncation=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

args = Seq2SeqTrainingArguments(
    output_dir="./game-t5-query-rewriter",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=3e-5,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    predict_with_generate=True,
    fp16=True,
    save_total_limit=2,
    logging_steps=50
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./game-t5-query-rewriter-final")
tokenizer.save_pretrained("./game-t5-query-rewriter-final")
```

------

# 13. 推理代码示例

训练好后，输入一个用户问题，输出多个改写 Query。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "./game-t5-query-rewriter-final"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def rewrite_query(user_query, intent="攻略类"):
    text = f"rewrite game rag query intent={intent}: {user_query}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    queries = [q.strip() for q in result.split(";") if q.strip()]
    return queries

if __name__ == "__main__":
    q = "澜现在怎么玩？"
    print(rewrite_query(q, intent="攻略类"))
```

输出可能是：

```text
[
  "澜 打法技巧",
  "澜 出装推荐",
  "澜 铭文搭配",
  "澜 连招技巧",
  "澜 打野思路"
]
```

------

# 14. 在 Java SpringBoot RAG 系统中怎么接入？

因为你的主系统是 Java SpringBoot，T5 模型建议单独部署成一个 Python 服务。

## 14.1 推荐架构

```text
SpringBoot RAG 主服务
   ↓ HTTP
Python FastAPI Query Rewrite 服务
   ↓
返回 rewritten_queries
   ↓
SpringBoot 调 Milvus / reranker / LLM
```

------

## 14.2 FastAPI 服务示例

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

model_path = "./game-t5-query-rewriter-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

class RewriteRequest(BaseModel):
    query: str
    intent: str = "攻略类"

class RewriteResponse(BaseModel):
    original_query: str
    rewritten_queries: list[str]

@app.post("/rewrite", response_model=RewriteResponse)
def rewrite(req: RewriteRequest):
    text = f"rewrite game rag query intent={req.intent}: {req.query}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=4,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    rewritten_queries = [q.strip() for q in result.split(";") if q.strip()]

    return RewriteResponse(
        original_query=req.query,
        rewritten_queries=rewritten_queries
    )
```

启动：

```bash
uvicorn app:app --host 0.0.0.0 --port 9001
```

------

## 14.3 SpringBoot 调用示例

```java
public class RewriteRequest {
    private String query;
    private String intent;

    public RewriteRequest(String query, String intent) {
        this.query = query;
        this.intent = intent;
    }

    public String getQuery() {
        return query;
    }

    public String getIntent() {
        return intent;
    }
}
public class RewriteResponse {
    private String originalQuery;
    private List<String> rewrittenQueries;

    public String getOriginalQuery() {
        return originalQuery;
    }

    public List<String> getRewrittenQueries() {
        return rewrittenQueries;
    }
}
@Service
public class QueryRewriteService {

    private final RestTemplate restTemplate = new RestTemplate();

    public List<String> rewrite(String query, String intent) {
        String url = "http://localhost:9001/rewrite";

        RewriteRequest request = new RewriteRequest(query, intent);

        RewriteResponse response = restTemplate.postForObject(
                url,
                request,
                RewriteResponse.class
        );

        if (response == null || response.getRewrittenQueries() == null) {
            return List.of(query);
        }

        List<String> result = new ArrayList<>();
        result.add(query); // 保留原始 Query
        result.addAll(response.getRewrittenQueries());

        return result;
    }
}
```

注意：一定要保留原始 Query，不要只用改写 Query。

------

# 15. 检索阶段怎么使用改写 Query？

假设 T5 输出：

```text
澜 打法技巧
澜 出装推荐
澜 铭文搭配
澜 连招技巧
```

你的检索可以这样做：

```text
原始 Query：澜现在怎么玩？
改写 Query 1：澜 打法技巧
改写 Query 2：澜 出装推荐
改写 Query 3：澜 铭文搭配
改写 Query 4：澜 连招技巧
```

每个 Query 都进入 Milvus：

```text
dense_vector_search(original_query)
dense_vector_search(rewritten_query_1)
dense_vector_search(rewritten_query_2)
dense_vector_search(rewritten_query_3)
dense_vector_search(rewritten_query_4)
```

如果你用 BGE-M3，还可以同时做：

```text
sparse_vector_search(rewritten_query_i)
```

最后用 RRF 融合：

```text
score(d) = Σ 1 / (k + rank_i(d))
```

这样可以避免某一个改写 Query 召回偏了。

------

# 16. 为什么不建议一开始用强化学习？

对于你的游戏 RAG 系统，**不建议一开始上 PPO**。

原因有四个：

第一，强化学习训练链路复杂，需要每次生成 Query 后走完整 RAG 流程，再根据答案质量给 reward。

第二，reward 难设计。游戏问答不一定有标准答案，EM/F1 不适合所有问题。

第三，训练成本高。每个样本都要检索、rerank、调用 LLM 或 Judge。

第四，工程调试困难。PPO 训练不稳定，容易生成奇怪 Query。

所以你的最佳路线是：

```text
先做 Prompt 改写
再做监督微调 T5
最后如果系统成熟，再考虑奖励优化
```

当前阶段只做监督微调就很合理。

------

# 17. 推荐落地路线

## 阶段一：不用训练，先用 prompt 生成训练数据

先用大模型生成伪标注数据：

```text
用户 Query → LLM 生成 rewritten queries
```

人工检查一部分，保证质量。

------

## 阶段二：微调 T5 Rewriter

训练：

```text
input: rewrite game rag query intent=攻略类: 澜怎么玩？
target: 澜 打法技巧; 澜 出装推荐; 澜 铭文搭配; 澜 连招技巧
```

得到：

```text
game-t5-query-rewriter
```

------

## 阶段三：接入 Java RAG 系统

```text
SpringBoot → FastAPI T5 Rewriter → Milvus → reranker → LLM
```

------

## 阶段四：离线评估

不要只看 T5 输出像不像人工改写，而要看检索效果。

建议评估：

| 指标             | 含义                                |
| ---------------- | ----------------------------------- |
| Recall@5         | 前 5 个召回 chunk 是否包含正确知识  |
| Recall@10        | 前 10 个召回 chunk 是否包含正确知识 |
| MRR              | 正确 chunk 排名是否靠前             |
| nDCG@10          | 相关 chunk 排序质量                 |
| Answer Accuracy  | 最终回答是否正确                    |
| Bad Rewrite Rate | 改写后检索变差的比例                |

重点比较：

```text
原始 Query 检索
vs
T5 改写 Query 检索
vs
原始 Query + T5 改写 Query 多路检索
```

通常第三种最好：

```text
原始 Query + 改写 Query 多路检索
```

------

# 18. 样本设计示例

下面给一组更接近游戏知识库的训练样本。

```json
{"input": "rewrite game rag query intent=攻略类: 孙悟空怎么秒人？", "target": "孙悟空 爆发连招; 孙悟空 秒人技巧; 孙悟空 出装铭文; 孙悟空 连招顺序"}
{"input": "rewrite game rag query intent=资料类: 李白大招怎么解锁？", "target": "李白 大招 解锁机制; 李白 青莲剑歌 释放条件; 李白 技能机制; 李白 普攻 解锁大招"}
{"input": "rewrite game rag query intent=攻略类: 妲己后期怎么打团？", "target": "妲己 后期团战技巧; 妲己 技能连招; 妲己 蹲草打法; 妲己 秒人思路"}
{"input": "rewrite game rag query intent=资讯类: 新赛季野区有什么变化？", "target": "新赛季 野区调整; 野区资源 改动; 打野机制 更新; 版本公告 野区变化"}
{"input": "rewrite game rag query intent=资料类: 影忍之足是什么装备？", "target": "影忍之足 装备效果; 影忍之足 属性介绍; 影忍之足 减伤机制; 影忍之足 适合英雄"}
{"input": "rewrite game rag query intent=攻略类: 后羿怎么打兰陵王？", "target": "后羿 对线 兰陵王; 后羿 防兰陵王技巧; 兰陵王 克制后羿; 后羿 生存出装"}
{"input": "rewrite game rag query intent=攻略类: 貂蝉怎么出装？", "target": "貂蝉 出装推荐; 貂蝉 铭文搭配; 貂蝉 冷却缩减装备; 貂蝉 法师出装"}
{"input": "rewrite game rag query intent=资料类: 反伤刺甲有什么用？", "target": "反伤刺甲 装备效果; 反伤刺甲 属性介绍; 反伤刺甲 反伤机制; 反伤刺甲 适合英雄"}
```

------

# 19. 你的系统中推荐的最终方案

我建议你采用这个版本：

```text
意图识别：BERT / MiniLM / Qwen 分类
Query 改写：监督微调 T5 / mT5
向量检索：BGE-M3 dense
稀疏检索：BGE-M3 sparse 或 BM25
融合：RRF
精排：bge-reranker-v2-m3
生成：豆包 / Qwen / DeepSeek / 本地大模型
```

完整链路：

```text
User Query
   ↓
Intent Classifier
   ↓
T5 Query Rewriter
   ↓
Original Query + Rewritten Queries
   ↓
Dense Search + Sparse Search
   ↓
RRF Fusion
   ↓
Rerank
   ↓
Prompt Assembly
   ↓
LLM Answer
```

------

# 20. 结论

可以使用，而且建议这样做：

```text
保留：T5 监督微调 Query Rewriter 思想
删除：PPO / 强化学习 / reward_fn / actor-critic
替换：Bing Search → Milvus / BGE-M3 混合检索
应用：游戏资料类、资讯类、攻略类 Query 改写
```

最推荐的落地方式是：

```text
先用大模型构造 3000～10000 条 Query 改写数据
再微调 mT5-base 或中文 T5-base
然后以 FastAPI 服务形式接入 SpringBoot RAG 系统
最后用 Recall@K、MRR、最终回答准确率评估效果
```

这样既继承了论文中 **Rewrite-Retrieve-Read** 的核心思想，又避免了强化学习带来的复杂度，非常适合作为你的游戏知识库 RAG 系统中的 Query 改写模块。
