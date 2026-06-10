下面给出一套**用 T5-large 770M 微调实现 RAG Query 改写**的完整方案。T5 是典型的 encoder-decoder 文本到文本模型，适合把“原始 Query / 多轮上下文”转换成“更适合检索的标准化 Query”。Hugging Face 的 `google-t5/t5-large` 标注为 770M 参数，T5 体系本身将 NLP 任务统一成 text-to-text 形式。([Hugging Face](https://huggingface.co/google-t5/t5-large?utm_source=chatgpt.com))

需要注意：原版 `google-t5/t5-large` 的模型卡列出的主要语言是 English、French、Romanian、German，不是专门中文模型。中文 RAG 场景可以复用下面流程，但更建议换成 **mT5-large、中文 T5、Mengzi-T5、CPM/T5 类中文生成模型**。([Hugging Face](https://huggingface.co/google-t5/t5-large/blame/main/README.md?utm_source=chatgpt.com))

------

# 一、Query 改写在 RAG 中的位置

RAG 原始流程一般是：

```text
用户原始问题
  ↓
Query 改写模型 T5-large
  ↓
标准化 Query / 多路 Query 扩展
  ↓
Milvus / Elasticsearch / 混合检索
  ↓
Reranker 精排
  ↓
LLM 生成答案
```

Query 改写的目标不是回答问题，而是把用户问题改成更适合检索的形式。

例如：

```text
原始 Query：
这个英雄被动咋触发？

多轮上下文：
前面用户正在问：王者荣耀 孙尚香 技能机制

改写后 Query：
王者荣耀 孙尚香 被动技能 触发条件
```

或者：

```text
原始 Query：
AD 早期诊断用 EEG 有啥特征？

改写后 Query：
阿尔茨海默病 早期诊断 EEG 特征 theta beta alpha 频段变化
```

------

# 二、训练数据怎么构造

T5 微调本质是监督学习，需要构造：

```text
输入：原始 Query + 可选上下文
输出：改写后的 Query
```

## 1. 单条改写格式

推荐先做单条改写。

### JSONL 数据示例

```json
{"input": "rewrite query for retrieval: query=这个英雄被动怎么触发？ history=用户正在询问王者荣耀孙尚香技能机制", "target": "王者荣耀 孙尚香 被动技能 触发条件"}
{"input": "rewrite query for retrieval: query=AD早期EEG有啥变化？ history=", "target": "阿尔茨海默病 早期诊断 EEG 频段特征 theta beta alpha 变化"}
{"input": "rewrite query for retrieval: query=这个模型怎么融合三种模态？ history=论文方法包含EEG PET sMRI", "target": "非配对多模态 EEG PET sMRI 阿尔茨海默病 诊断 融合方法"}
```

## 2. 多条 Query 扩展格式

如果想做 Multi-Query Retrieval，可以让 T5 一次生成多条检索 Query。

```json
{
  "input": "generate retrieval queries: query=AD早期EEG有啥变化？ history=",
  "target": "阿尔茨海默病 早期诊断 EEG 频段特征; AD EEG theta beta alpha 变化; 阿尔茨海默病 脑电信号 特征 提取"
}
```

推理时用 `;` 分割，分别检索，然后用 RRF 融合。

------

# 三、数据来源

实际项目中可以从以下几类数据构造训练集。

## 1. 人工标注数据

质量最高，适合做验证集和测试集。

格式：

```text
原始问题 → 改写问题
```

例如：

```text
他的被动是什么
→ 王者荣耀 孙尚香 被动技能 效果
```

## 2. 基于知识库标题构造

如果你的知识库 chunk 有字段：

```text
title
content
doc_id
chunk_id
```

可以根据 `title + chunk标题 + 用户口语问题` 构造训练样本。

例如知识库 chunk：

```text
title: 孙尚香技能介绍
chunk: 被动技能：活力迸发
content: 孙尚香每次普通攻击命中敌人后，会减少翻滚突袭的冷却时间。
```

构造：

```text
input:
rewrite query for retrieval: query=她普攻以后技能会变快吗？ history=当前角色是孙尚香

target:
王者荣耀 孙尚香 普通攻击 减少 翻滚突袭 冷却 被动技能
```

## 3. 用大模型生成伪标注

如果没有人工数据，可以用 GPT / Qwen / Doubao 等生成初始训练集。

提示词示例：

```text
你是RAG系统的Query改写标注员。
请把用户问题改写成适合向量检索和关键词检索的标准Query。
要求：
1. 保留实体、属性、约束条件。
2. 不回答问题。
3. 不添加知识库中不存在的事实。
4. 输出一条短Query。

用户问题：{query}
多轮上下文：{history}
知识库相关标题：{doc_title}
输出：
```

然后对生成结果做过滤：

```text
原 Query 检索 Recall@10
改写 Query 检索 Recall@10
```

只保留改写后检索效果更好的样本。

------

# 四、推荐数据规模

| 阶段     | 数据量             | 说明                           |
| -------- | ------------------ | ------------------------------ |
| 快速验证 | 1,000 - 5,000 条   | 能看出模型是否学到格式         |
| 可用版本 | 10,000 - 50,000 条 | 适合一个垂直领域 RAG           |
| 稳定版本 | 100,000 条以上     | 多领域、多意图、多轮改写更稳定 |

如果你的 RAG 是游戏知识库、医学论文知识库、项目文档知识库，建议先人工精标 500 - 1000 条，再用大模型扩展到 1 万条以上。

------

# 五、训练输入模板设计

推荐统一使用 prefix，因为 T5 对任务前缀比较敏感。

## 单轮 Query 改写

```text
rewrite query for retrieval: query={用户原始问题}
```

## 多轮 Query 改写

```text
rewrite query for retrieval: history={最近2-4轮对话} query={当前问题}
```

## 带领域信息

```text
rewrite query for retrieval: domain=王者荣耀 knowledge_base=英雄技能 query={用户问题}
```

## 带实体抽取结果

```text
rewrite query for retrieval: entity=孙尚香 intent=技能机制 query=她的被动怎么触发？
```

推荐最终使用：

```text
rewrite query for retrieval: domain={领域} history={历史对话} query={当前问题}
```

------

# 六、项目目录结构

```text
t5_query_rewrite/
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── outputs/
│   └── t5-large-query-rewrite/
├── train_t5_rewrite.py
├── infer_t5_rewrite.py
├── evaluate_retrieval.py
└── requirements.txt
```

------

# 七、安装依赖

```bash
pip install torch transformers datasets sentencepiece accelerate evaluate peft
```

`transformers` 官方支持 T5 这类 encoder-decoder 模型，并提供训练接口。Hugging Face 文档也将 T5 描述为把任务统一为 text-to-text 的模型。([Hugging Face](https://huggingface.co/docs/transformers/en/training?utm_source=chatgpt.com))

------

# 八、训练脚本：全参数微调版本

文件：`train_t5_rewrite.py`

```python
import os
import argparse
from datasets import load_dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google-t5/t5-large")
    parser.add_argument("--train_file", type=str, default="data/train.jsonl")
    parser.add_argument("--dev_file", type=str, default="data/dev.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/t5-large-query-rewrite")
    parser.add_argument("--max_source_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    return parser.parse_args()

def main():
    args = parse_args()

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.dev_file
        }
    )

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    def preprocess(batch):
        inputs = batch["input"]
        targets = batch["target"]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True
        )

        labels = tokenizer(
            text_target=targets,
            max_length=args.max_target_length,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
```

运行：

```bash
python train_t5_rewrite.py \
  --model_name google-t5/t5-large \
  --train_file data/train.jsonl \
  --dev_file data/dev.jsonl \
  --output_dir outputs/t5-large-query-rewrite \
  --batch_size 2 \
  --grad_accum 8 \
  --epochs 3 \
  --lr 3e-5
```

------

# 九、推荐：LoRA 微调版本

T5-large 有 770M 参数，全参数微调显存压力比较大。实际项目更推荐 LoRA。

优点：

```text
1. 显存更低；
2. 训练更快；
3. 便于保存多个领域版本；
4. 不容易过拟合小数据集。
```

核心修改：

```python
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q", "v"]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

推荐 LoRA 参数：

| 参数           | 建议值           |
| -------------- | ---------------- |
| r              | 8 或 16          |
| lora_alpha     | 16 或 32         |
| lora_dropout   | 0.05             |
| target_modules | `["q", "v"]`     |
| learning_rate  | `1e-4` 到 `3e-4` |
| epoch          | 3 - 5            |

LoRA 训练时学习率可以比全参数微调更大。

------

# 十、推理脚本

文件：`infer_t5_rewrite.py`

```python
import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/t5-large-query-rewrite")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--history", type=str, default="")
    parser.add_argument("--domain", type=str, default="general")
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    return parser.parse_args()

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = T5Tokenizer.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    input_text = (
        f"rewrite query for retrieval: "
        f"domain={args.domain} "
        f"history={args.history} "
        f"query={args.query}"
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False
        )

    rewritten_query = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print(rewritten_query)

if __name__ == "__main__":
    main()
```

运行：

```bash
python infer_t5_rewrite.py \
  --model_dir outputs/t5-large-query-rewrite \
  --domain "game_rag" \
  --history "用户正在询问王者荣耀孙尚香技能机制" \
  --query "她的被动怎么触发" \
  --num_beams 1
```

输出示例：

```text
王者荣耀 孙尚香 被动技能 触发条件
```

------

# 十一、接入 RAG 检索流程

推荐接入方式：

```python
raw_query = "她的被动怎么触发"
history = "用户正在询问王者荣耀孙尚香技能机制"

rewrite_query = t5_rewrite(raw_query, history)

retrieval_queries = [
    raw_query,
    rewrite_query
]

docs = []
for q in retrieval_queries:
    docs.extend(vector_search(q))
    docs.extend(keyword_search(q))

docs = rrf_fusion(docs)
docs = rerank(raw_query, docs)
answer = llm_generate(raw_query, docs)
```

不要只用改写后的 Query，建议同时保留原始 Query：

```text
原始 Query + 改写 Query 一起检索
```

这样可以避免模型改写错误导致召回丢失。

------

# 十二、训练评估指标

不要只看 BLEU、ROUGE，因为 Query 改写最终服务于检索。

建议评估：

| 指标                 | 含义                       |
| -------------------- | -------------------------- |
| Rewrite Exact Match  | 改写结果是否和标注完全一致 |
| BLEU / ROUGE         | 文本相似度，仅作辅助       |
| Recall@5 / Recall@10 | 改写后能否召回正确文档     |
| MRR@10               | 正确文档排名是否靠前       |
| nDCG@10              | 多相关文档排序质量         |
| RAG Answer Accuracy  | 最终答案是否更准           |

最重要的是：

```text
原始 Query 检索 Recall@10
vs
改写 Query 检索 Recall@10
vs
原始 Query + 改写 Query 融合检索 Recall@10
```

如果第三种最好，说明改写模型有效。

------

# 十三、超参数建议

## 全参数微调

| 参数                        | 建议              |
| --------------------------- | ----------------- |
| max_source_length           | 128 或 256        |
| max_target_length           | 32 或 64          |
| batch_size                  | 1 - 4             |
| gradient_accumulation_steps | 8 - 32            |
| learning_rate               | `1e-5` 到 `5e-5`  |
| epoch                       | 2 - 5             |
| fp16                        | 开启              |
| num_beams                   | 训练评估时 1 或 4 |

## LoRA 微调

| 参数           | 建议             |
| -------------- | ---------------- |
| r              | 8 / 16           |
| lora_alpha     | 16 / 32          |
| learning_rate  | `1e-4` 到 `3e-4` |
| batch_size     | 4 - 16           |
| epoch          | 3 - 5            |
| target_modules | `q`, `v`         |

------

# 十四、显存需求大概多少

T5-large 770M 全参数微调比较吃显存。

| 方式                         | 建议显存             |
| ---------------------------- | -------------------- |
| 推理 FP16                    | 8GB - 12GB 可尝试    |
| LoRA 微调                    | 16GB 比较合适        |
| 全参数微调                   | 24GB 起步，32GB 更稳 |
| 长输入 + 大 batch 全参数微调 | 40GB 以上更稳        |

如果你只有 12GB 显存，建议：

```text
LoRA + fp16 + batch_size=1/2 + gradient_accumulation
```

------

# 十五、改写一条 Query 大概多长时间

这个时间主要由以下因素决定：

```text
1. 硬件：CPU / 3060 / 3090 / 4090 / A100
2. 输入长度：history 是否很长
3. 输出长度：max_new_tokens 是 32 还是 64
4. 解码方式：greedy 比 beam search 快
5. 是否批处理：batch 推理平均单条更快
6. 是否常驻服务：模型是否已经加载到显存
```

以常见设置为例：

```text
输入长度：128 - 256 tokens
输出长度：32 - 64 tokens
num_beams=1 或 4
模型已加载，不包含冷启动时间
```

大概耗时如下：

| 环境            | greedy，num_beams=1 | beam search，num_beams=4 |
| --------------- | ------------------- | ------------------------ |
| CPU 8 核        | 1.5 - 6 秒 / 条     | 4 - 15 秒 / 条           |
| RTX 3060 12GB   | 100 - 300 ms / 条   | 300 - 800 ms / 条        |
| RTX 3090 / 4090 | 30 - 120 ms / 条    | 120 - 400 ms / 条        |
| A10 / A100      | 10 - 80 ms / 条     | 50 - 200 ms / 条         |

实际 RAG 系统中，建议使用：

```text
num_beams=1
max_new_tokens=32
模型常驻 GPU
批量处理请求
```

这样在 3090 / 4090 / A10 这类 GPU 上，**单条 Query 改写通常可以控制在几十毫秒到一两百毫秒**。如果用 CPU，T5-large 会明显拖慢 RAG，一条 Query 可能需要几秒。

------

# 十六、上线建议

实际工程里可以把 T5 改写模型做成单独服务：

```text
Java SpringBoot RAG 服务
  ↓ HTTP/gRPC
Python FastAPI T5 Rewrite 服务
  ↓
返回 rewrite_query
```

Python 服务示例结构：

```text
POST /rewrite

输入：
{
  "query": "她的被动怎么触发",
  "history": "用户正在询问王者荣耀孙尚香技能机制",
  "domain": "game_rag"
}

输出：
{
  "rewrite_query": "王者荣耀 孙尚香 被动技能 触发条件"
}
```

部署时注意：

```text
1. 模型启动时加载一次，不要每次请求都加载。
2. max_new_tokens 控制在 32 或 64。
3. num_beams 默认为 1。
4. 改写失败时回退到原始 Query。
5. 检索时使用 原始 Query + 改写 Query 融合。
```

------

# 十七、最终推荐方案

如果你的目标是做一个稳定的 RAG Query 改写模块，我建议这样做：

```text
第一阶段：
人工构造 1000 条高质量 query → rewrite 数据，先验证流程。

第二阶段：
用大模型生成 1 万 - 5 万条伪标注数据，再人工抽检。

第三阶段：
使用 LoRA 微调 T5-large 或 mT5-large。

第四阶段：
评估原始 Query、改写 Query、融合 Query 的 Recall@10 和 MRR@10。

第五阶段：
上线时使用 greedy decoding，max_new_tokens=32，并保留原始 Query 兜底。
```

对于中文 RAG，模型选择建议是：

```text
优先：mT5-large / 中文 T5 / 中文指令微调 seq2seq 模型
可用但不优先：google-t5/t5-large
```

如果只是英文知识库或英文 Query 改写，`google-t5/t5-large` 更合适。