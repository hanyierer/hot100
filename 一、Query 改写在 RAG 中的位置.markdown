可以，直接补你现在这个类里的 `sparseVectorSearch()`。你当前代码中稀疏检索方法只是占位，`sparseVectorSearch` 直接返回空列表，所以只要把这一部分替换掉，并补充少量配置即可。

下面按你当前 Milvus 字段设计来实现：

```text
id
title
chunk_id
content
content_dense_vector
content_sparse_vector
doc_id
```

其中：

```text
content_dense_vector  → 稠密向量字段
content_sparse_vector → 稀疏向量字段
```

------

# 一、先说明两种稀疏检索方式

Milvus 稀疏检索有两种常见做法：

| 方式                              | 查询时传什么         | 适合场景                                |
| --------------------------------- | -------------------- | --------------------------------------- |
| Milvus BM25 Function 自动稀疏向量 | 直接传原始文本 Query | 适合关键词检索、全文检索                |
| BGE-M3 / SPLADE 生成稀疏向量      | 传 `SparseFloatVec`  | 适合模型生成的 learned sparse retrieval |

你之前说的是“Milvus 自动生成稀疏向量”，所以这里优先采用 **Milvus BM25 Function + EmbeddedText** 的方式。

Milvus 的 BM25 Function 会把文本字段转换成稀疏向量，并且查询时也可以直接传入原始文本，Milvus 会自动把查询文本转换成稀疏向量进行检索。官方 Java 示例里也是用 `new EmbeddedText("query text")` 作为查询数据。([Milvus](https://milvus.io/docs/full-text-search.md))

------

# 二、需要补充的 import

在你的 `MilvusVectorRetrievalService` 顶部增加：

```java
import io.milvus.v2.service.vector.request.data.EmbeddedText;
```

如果你后续想用 BGE-M3 自己生成稀疏向量，还需要：

```java
import io.milvus.v2.service.vector.request.data.SparseFloatVec;
```

但本次 BM25 自动稀疏检索只需要 `EmbeddedText`。

------

# 三、替换 sparseVectorSearch 方法

把你原来的这个方法：

```java
private List<SearchHit> sparseVectorSearch(String query) {
    long start = System.currentTimeMillis();
    log.debug("稀疏向量检索暂未实现，返回空列表，query='{}'，耗时: {} ms", query, System.currentTimeMillis() - start);
    return List.of();
}
```

替换成下面完整实现：

```java
/**
 * 稀疏向量检索。
 *
 * <p>适用于 Milvus 内置 BM25 Function 自动生成的稀疏向量字段。
 * <p>查询时不需要自己生成 sparse vector，而是直接传入 EmbeddedText，
 * Milvus 会使用和入库时相同的 Analyzer + BM25 Function 将查询文本转换为稀疏向量。
 *
 * @param query 原始查询文本
 * @return 稀疏检索结果列表
 */
private List<SearchHit> sparseVectorSearch(String query) {
    long methodStart = System.currentTimeMillis();

    if (query == null || query.isBlank()) {
        return List.of();
    }

    try {
        String collectionName = properties.getMilvus().getCollection();
        String sparseField = properties.getMilvus().getSparseVectorField();

        if (sparseField == null || sparseField.isBlank()) {
            log.warn("稀疏检索失败：未配置 sparseVectorField");
            return List.of();
        }

        /*
         * 注意：
         * 如果 content_sparse_vector 是 BM25 Function 自动生成的字段，
         * outputFields 里不要包含 content_sparse_vector 本身。
         *
         * Milvus 对 BM25 生成的 sparse 字段通常不允许直接输出。
         */
        List<String> outputFields = buildSparseOutputFields();

        /*
         * BM25 稀疏检索搜索参数。
         *
         * drop_ratio_search 表示搜索时忽略一部分低权重词项。
         * 如果你不确定 SDK 版本是否支持该参数，可以先用空 Map。
         */
        Map<String, Object> searchParams = new HashMap<>();
        Double dropRatioSearch = properties.getMilvus().getSparseDropRatioSearch();
        if (dropRatioSearch != null && dropRatioSearch > 0) {
            searchParams.put("drop_ratio_search", dropRatioSearch);
        }

        SearchReq searchReq = SearchReq.builder()
                .collectionName(collectionName)
                .annsField(sparseField)
                .data(Collections.singletonList(new EmbeddedText(query)))
                .topK(properties.getMilvus().getSparseTopK())
                .outputFields(outputFields)
                .consistencyLevel(ConsistencyLevel.valueOf(properties.getMilvus().getConsistencyLevel()))
                .searchParams(searchParams)
                .build();

        long searchStart = System.currentTimeMillis();
        SearchResp response = milvusClient.search(searchReq);
        long searchCost = System.currentTimeMillis() - searchStart;

        long parseStart = System.currentTimeMillis();
        List<SearchHit> hits = parseSearchResp(response, RecallChannel.SPARSE_VECTOR);
        long parseCost = System.currentTimeMillis() - parseStart;

        long totalCost = System.currentTimeMillis() - methodStart;
        log.info("稀疏检索总耗时: {} ms (Milvus搜索: {}ms, 解析: {}ms)，命中数: {}",
                totalCost, searchCost, parseCost, hits.size());

        return hits;
    } catch (Exception e) {
        log.warn("稀疏向量检索失败, query='{}': {}", query, e.getMessage(), e);
        return List.of();
    }
}
```

------

# 四、增加 buildSparseOutputFields 方法

把下面这个辅助方法加到 `MilvusVectorRetrievalService` 里，建议放在 `parseSearchResp()` 前面或后面都可以。

```java
/**
 * 构建稀疏检索的输出字段。
 *
 * <p>如果 content_sparse_vector 是 Milvus BM25 Function 自动生成字段，
 * 不建议放入 outputFields，否则部分 Milvus 版本会报错。
 *
 * @return 稀疏检索可以安全返回的字段列表
 */
private List<String> buildSparseOutputFields() {
    List<String> configuredFields = properties.getMilvus().getOutputFields();
    if (configuredFields == null || configuredFields.isEmpty()) {
        return List.of(
                properties.getMilvus().getPrimaryKeyField(),
                properties.getMilvus().getTitleField(),
                properties.getMilvus().getChunkIdField(),
                properties.getMilvus().getTextField()
        );
    }

    String sparseField = properties.getMilvus().getSparseVectorField();

    List<String> outputFields = new ArrayList<>();
    for (String field : configuredFields) {
        if (field == null || field.isBlank()) {
            continue;
        }

        /*
         * BM25 自动生成的 sparse vector 字段不能作为普通字段输出。
         */
        if (field.equals(sparseField)) {
            continue;
        }

        outputFields.add(field);
    }

    return TextUtils.distinctNonBlank(outputFields, outputFields.size());
}
```

------

# 五、RagProperties 需要补充的配置字段

你的代码里已经有：

```java
properties.getMilvus().isSparseSearchEnabled()
```

说明配置类里大概率已经有 `sparseSearchEnabled`。还需要补充这些字段：

```java
private String sparseVectorField;
private Integer sparseTopK;
private Double sparseDropRatioSearch;
```

示例：

```java
@Data
@ConfigurationProperties(prefix = "rag")
public class RagProperties {

    private Milvus milvus = new Milvus();

    @Data
    public static class Milvus {

        private String endpoint;
        private String token;
        private String database;
        private String collection;

        private String primaryKeyField = "id";
        private String titleField = "title";
        private String chunkIdField = "chunk_id";
        private String textField = "content";
        private String docIdField = "doc_id";

        private String denseVectorField = "content_dense_vector";
        private String sparseVectorField = "content_sparse_vector";

        private boolean denseSearchEnabled = true;
        private boolean sparseSearchEnabled = true;

        private Integer denseTopK = 20;
        private Integer sparseTopK = 20;

        private Double sparseDropRatioSearch = 0.0;

        private String consistencyLevel = "BOUNDED";

        private List<String> outputFields = List.of(
                "id",
                "title",
                "chunk_id",
                "content",
                "doc_id"
        );
    }
}
```

------

# 六、application.yml 配置示例

你可以这样配置：

```yaml
rag:
  milvus:
    endpoint: http://localhost:19530
    token: root:Milvus
    database: default
    collection: game_knowledge

    primary-key-field: id
    title-field: title
    chunk-id-field: chunk_id
    text-field: content
    doc-id-field: doc_id

    dense-vector-field: content_dense_vector
    sparse-vector-field: content_sparse_vector

    dense-search-enabled: true
    sparse-search-enabled: true

    dense-top-k: 20
    sparse-top-k: 20

    sparse-drop-ratio-search: 0.0

    consistency-level: BOUNDED

    output-fields:
      - id
      - title
      - chunk_id
      - content
      - doc_id
```

注意：不要把下面这个字段放到 `output-fields` 里：

```yaml
- content_sparse_vector
```

因为如果它是 BM25 Function 自动生成的稀疏字段，Milvus 通常不允许直接输出该字段。官方文档也说明 BM25 Function 生成的 sparse 字段不能作为普通输出字段返回。([Milvus](https://milvus.io/docs/full-text-search.md))

------

# 七、完整修改后的关键代码片段

你的 `MilvusVectorRetrievalService` 主要改成这样。

## 1. import 部分增加

```java
import io.milvus.v2.service.vector.request.data.EmbeddedText;
```

------

## 2. sparseVectorSearch 完整实现

```java
private List<SearchHit> sparseVectorSearch(String query) {
    long methodStart = System.currentTimeMillis();

    if (query == null || query.isBlank()) {
        return List.of();
    }

    try {
        String collectionName = properties.getMilvus().getCollection();
        String sparseField = properties.getMilvus().getSparseVectorField();

        if (sparseField == null || sparseField.isBlank()) {
            log.warn("稀疏检索失败：未配置 sparseVectorField");
            return List.of();
        }

        List<String> outputFields = buildSparseOutputFields();

        Map<String, Object> searchParams = new HashMap<>();
        Double dropRatioSearch = properties.getMilvus().getSparseDropRatioSearch();
        if (dropRatioSearch != null && dropRatioSearch > 0) {
            searchParams.put("drop_ratio_search", dropRatioSearch);
        }

        SearchReq searchReq = SearchReq.builder()
                .collectionName(collectionName)
                .annsField(sparseField)
                .data(Collections.singletonList(new EmbeddedText(query)))
                .topK(properties.getMilvus().getSparseTopK())
                .outputFields(outputFields)
                .consistencyLevel(ConsistencyLevel.valueOf(properties.getMilvus().getConsistencyLevel()))
                .searchParams(searchParams)
                .build();

        long searchStart = System.currentTimeMillis();
        SearchResp response = milvusClient.search(searchReq);
        long searchCost = System.currentTimeMillis() - searchStart;

        long parseStart = System.currentTimeMillis();
        List<SearchHit> hits = parseSearchResp(response, RecallChannel.SPARSE_VECTOR);
        long parseCost = System.currentTimeMillis() - parseStart;

        long totalCost = System.currentTimeMillis() - methodStart;
        log.info("稀疏检索总耗时: {} ms (Milvus搜索: {}ms, 解析: {}ms)，命中数: {}",
                totalCost, searchCost, parseCost, hits.size());

        return hits;
    } catch (Exception e) {
        log.warn("稀疏向量检索失败, query='{}': {}", query, e.getMessage(), e);
        return List.of();
    }
}
```

------

## 3. buildSparseOutputFields 辅助方法

```java
private List<String> buildSparseOutputFields() {
    List<String> configuredFields = properties.getMilvus().getOutputFields();
    if (configuredFields == null || configuredFields.isEmpty()) {
        return List.of(
                properties.getMilvus().getPrimaryKeyField(),
                properties.getMilvus().getTitleField(),
                properties.getMilvus().getChunkIdField(),
                properties.getMilvus().getTextField()
        );
    }

    String sparseField = properties.getMilvus().getSparseVectorField();

    List<String> outputFields = new ArrayList<>();
    for (String field : configuredFields) {
        if (field == null || field.isBlank()) {
            continue;
        }
        if (field.equals(sparseField)) {
            continue;
        }
        outputFields.add(field);
    }

    return TextUtils.distinctNonBlank(outputFields, outputFields.size());
}
```

------

# 八、Milvus Collection 必须满足的条件

你的 Milvus Collection 需要这样设计：

```text
content 字段：
    DataType.VarChar
    enable_analyzer = true

content_sparse_vector 字段：
    DataType.SparseFloatVector

BM25 Function：
    input_field_names = ["content"]
    output_field_names = ["content_sparse_vector"]
```

Milvus 文档中要求 BM25 全文检索至少包含文本字段、稀疏向量字段和 BM25 Function；文本字段需要启用 analyzer，BM25 Function 会把文本转成稀疏向量。([Milvus](https://milvus.io/docs/full-text-search.md))

示例 Collection Schema：

```java
import io.milvus.common.clientenum.FunctionType;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq.Function;

import java.util.*;

CreateCollectionReq.CollectionSchema schema = CreateCollectionReq.CollectionSchema.builder()
        .enableDynamicField(false)
        .build();

schema.addField(AddFieldReq.builder()
        .fieldName("id")
        .dataType(DataType.VarChar)
        .isPrimaryKey(true)
        .maxLength(64)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("title")
        .dataType(DataType.VarChar)
        .maxLength(512)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("chunk_id")
        .dataType(DataType.VarChar)
        .maxLength(64)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("content")
        .dataType(DataType.VarChar)
        .maxLength(65535)
        .enableAnalyzer(true)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("doc_id")
        .dataType(DataType.VarChar)
        .maxLength(64)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("content_dense_vector")
        .dataType(DataType.FloatVector)
        .dimension(1024)
        .build());

schema.addField(AddFieldReq.builder()
        .fieldName("content_sparse_vector")
        .dataType(DataType.SparseFloatVector)
        .build());

schema.addFunction(Function.builder()
        .functionType(FunctionType.BM25)
        .name("content_bm25_func")
        .inputFieldNames(Collections.singletonList("content"))
        .outputFieldNames(Collections.singletonList("content_sparse_vector"))
        .build());
```

稀疏向量字段不需要设置维度，Milvus 支持 `SPARSE_FLOAT_VECTOR`，并且稀疏向量字段可以和稠密向量字段放在同一个 Collection 中做混合检索。([Milvus](https://milvus.io/docs/sparse_vector.md))

------

# 九、稀疏向量索引配置

BM25 Function 对应的稀疏字段需要建索引。

示例：

```java
Map<String, Object> sparseIndexParams = new HashMap<>();
sparseIndexParams.put("inverted_index_algo", "DAAT_MAXSCORE");
sparseIndexParams.put("bm25_k1", 1.2);
sparseIndexParams.put("bm25_b", 0.75);

List<IndexParam> indexes = new ArrayList<>();

indexes.add(IndexParam.builder()
        .fieldName("content_sparse_vector")
        .indexType(IndexParam.IndexType.AUTOINDEX)
        .metricType(IndexParam.MetricType.BM25)
        .extraParams(sparseIndexParams)
        .build());
```

如果你不是 BM25 Function，而是自己插入 BGE-M3 / SPLADE 生成的 sparse vector，那么稀疏索引通常使用：

```text
SPARSE_INVERTED_INDEX
```

或者：

```text
SPARSE_WAND
```

手动 sparse vector 场景下，Milvus 稀疏向量目前主要支持 `IP` 作为相似度度量；BM25 Function 场景下则使用 `BM25` metric。([Milvus](https://milvus.io/docs/sparse_vector.md))

------

# 十、如果你不是 BM25，而是 BGE-M3 生成 sparse vector

如果你离线入库时的 `content_sparse_vector` 不是 Milvus BM25 Function 自动生成，而是 BGE-M3 生成的稀疏向量，那么上面的 `EmbeddedText` 不能用。

这时应该这样写：

```java
import io.milvus.v2.service.vector.request.data.SparseFloatVec;
import java.util.SortedMap;
import java.util.TreeMap;
```

实现方式：

```java
private List<SearchHit> sparseVectorSearch(String query) {
    long methodStart = System.currentTimeMillis();

    if (query == null || query.isBlank()) {
        return List.of();
    }

    try {
        long embedStart = System.currentTimeMillis();

        /*
         * 这里需要你自己实现 bgeSparseEmbeddingClient.sparseEmbed(query)
         * 返回格式必须是 SortedMap<Long, Float>
         *
         * 例如：
         * {
         *   101L: 0.83f,
         *   20345L: 0.41f,
         *   99871L: 0.12f
         * }
         */
        SortedMap<Long, Float> sparseVector = bgeSparseEmbeddingClient.sparseEmbed(query);

        long embedCost = System.currentTimeMillis() - embedStart;

        if (sparseVector == null || sparseVector.isEmpty()) {
            log.warn("BGE-M3 稀疏向量为空，query='{}'", query);
            return List.of();
        }

        Map<String, Object> searchParams = new HashMap<>();
        Double dropRatioSearch = properties.getMilvus().getSparseDropRatioSearch();
        if (dropRatioSearch != null && dropRatioSearch > 0) {
            searchParams.put("drop_ratio_search", dropRatioSearch);
        }

        SearchReq searchReq = SearchReq.builder()
                .collectionName(properties.getMilvus().getCollection())
                .annsField(properties.getMilvus().getSparseVectorField())
                .data(Collections.singletonList(new SparseFloatVec(sparseVector)))
                .topK(properties.getMilvus().getSparseTopK())
                .outputFields(buildSparseOutputFields())
                .consistencyLevel(ConsistencyLevel.valueOf(properties.getMilvus().getConsistencyLevel()))
                .searchParams(searchParams)
                .build();

        long searchStart = System.currentTimeMillis();
        SearchResp response = milvusClient.search(searchReq);
        long searchCost = System.currentTimeMillis() - searchStart;

        List<SearchHit> hits = parseSearchResp(response, RecallChannel.SPARSE_VECTOR);

        log.info("BGE-M3 稀疏检索总耗时: {} ms (SparseEmbedding: {}ms, Milvus搜索: {}ms)，命中数: {}",
                System.currentTimeMillis() - methodStart,
                embedCost,
                searchCost,
                hits.size());

        return hits;
    } catch (Exception e) {
        log.warn("BGE-M3 稀疏向量检索失败, query='{}': {}", query, e.getMessage(), e);
        return List.of();
    }
}
```

Milvus Java SDK 的多向量检索示例中，`SparseFloatVec` 构造参数就是 `SortedMap`，用于 `SparseFloatVector` 字段检索。([Milvus](https://milvus.io/api-reference/java/v2.4.x/v2/Vector/hybridSearch.md))

------

# 十一、你当前项目推荐采用的最终版本

结合你之前的字段描述：

```text
content_dense_vector
content_sparse_vector
```

以及你说的“自动生成 Milvus 稀疏向量”，推荐你用这个版本：

```java
.data(Collections.singletonList(new EmbeddedText(query)))
.annsField(properties.getMilvus().getSparseVectorField())
```

也就是：

```text
原始 Query 文本
    ↓
EmbeddedText
    ↓
Milvus Analyzer
    ↓
BM25 Function
    ↓
content_sparse_vector 检索
    ↓
返回关键词相关结果
```

最终召回阶段就是：

```text
denseVectorSearch(query)
    → 语义召回

sparseVectorSearch(query)
    → 关键词 / 精确词召回

RRF 融合
    → 综合排序

Reranker 精排
    → 最终 topK
```

这样你的 RAG 检索就同时具备：

```text
稠密检索：解决语义相似问题
稀疏检索：解决关键词、专有名词、技能名、装备名、版本号命中问题
```

例如：

```text
“孙尚香一技能”
“影刃属性”
“S36 赛季更新”
“红 buff 刷新时间”
```

这类 Query，稀疏检索通常会比单纯 dense 检索更稳。下面给出一套**用 T5-large 770M 微调实现 RAG Query 改写**的完整方案。T5 是典型的 encoder-decoder 文本到文本模型，适合把“原始 Query / 多轮上下文”转换成“更适合检索的标准化 Query”。Hugging Face 的 `google-t5/t5-large` 标注为 770M 参数，T5 体系本身将 NLP 任务统一成 text-to-text 形式。([Hugging Face](https://huggingface.co/google-t5/t5-large?utm_source=chatgpt.com))

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

如果只是英文知识库或英文 Query 改写，`google-t5/t5-large` 更合适。下面给出一个**微调 BERT 实现 RAG 系统意图识别**的完整方案，面向你之前的 RAG 在线检索流程：

> 用户 Query → 意图识别 → Query 改写/扩展 → 检索策略路由 → Milvus / 关键词 / 资讯库召回 → Rerank → LLM 生成

这里将意图分为三类：

| 意图类别 | 含义                                                 | 示例 Query                                     | RAG 策略                         |
| -------- | ---------------------------------------------------- | ---------------------------------------------- | -------------------------------- |
| 资料类   | 查固定知识、角色属性、技能说明、装备说明、背景设定等 | “孙尚香一技能是什么？”、“影刃属性是什么？”     | 查静态知识库，重视准确性         |
| 资讯类   | 查最新活动、版本更新、公告、新闻、赛季变化等         | “最新版本更新了什么？”、“这个赛季有什么活动？” | 查资讯库，按时间过滤，重视时效性 |
| 攻略类   | 查玩法、出装、连招、阵容、技巧、打法等               | “李白怎么玩？”、“打野开局路线怎么选？”         | 查攻略知识，召回更多步骤型内容   |

------

# 一、整体方案

## 1. 模块位置

在 RAG 系统中，BERT 意图识别模块应该放在 **Query 理解阶段的最前面**。

完整流程如下：

```text
用户 Query
   ↓
BERT 意图识别
   ↓
根据意图选择不同 Query 处理方式
   ↓
Query 改写 / 扩展 / 关键词提取
   ↓
根据意图选择不同检索策略
   ↓
Milvus 向量召回 / 关键词召回 / 资讯库召回
   ↓
Reranker 精排
   ↓
LLM 生成最终回答
```

意图识别结果可以设计成：

```json
{
  "query": "李白怎么玩？",
  "intent": "guide",
  "intent_name": "攻略类",
  "confidence": 0.9821
}
```

------

# 二、意图类别设计

建议内部使用英文标签，方便代码维护。

```text
资料类：document
资讯类：news
攻略类：guide
```

也可以设计成枚举：

```java
public enum QueryIntent {
    DOCUMENT, // 资料类
    NEWS,     // 资讯类
    GUIDE,    // 攻略类
    UNKNOWN   // 低置信度兜底
}
```

------

# 三、数据集构建

## 1. 数据格式

建议使用 CSV 或 JSONL。

CSV 格式：

```csv
query,label
孙尚香一技能是什么,document
影刃这件装备有什么属性,document
王者荣耀最新版本更新了什么,news
这个赛季有哪些新活动,news
李白怎么玩,guide
韩信打野路线怎么选,guide
```

JSONL 格式：

```json
{"query": "孙尚香一技能是什么", "label": "document"}
{"query": "王者荣耀最新版本更新了什么", "label": "news"}
{"query": "李白怎么玩", "label": "guide"}
```

------

## 2. 每类数据量建议

如果是初期可用版本：

| 数据规模          | 效果                 |
| ----------------- | -------------------- |
| 每类 300 条       | 可以跑通，但泛化一般 |
| 每类 1000 条      | 基本可用             |
| 每类 3000 条以上  | 效果比较稳定         |
| 每类 10000 条以上 | 适合正式上线         |

对于你的 RAG 系统，建议初期至少准备：

```text
资料类：1000 条
资讯类：1000 条
攻略类：1000 条
总计：3000 条左右
```

数据划分：

```text
训练集：80%
验证集：10%
测试集：10%
```

------

## 3. 三类样本示例

### 资料类 document

这类 Query 通常是在问固定知识。

```text
孙权的技能是什么
孙尚香的一技能有什么效果
影刃的装备属性是什么
破军适合哪些英雄
王者荣耀里蓝 buff 有什么作用
红 buff 的刷新时间是多少
刘备的被动技能介绍
小乔的大招叫什么
防御塔机制是什么
主宰和暴君有什么区别
```

### 资讯类 news

这类 Query 通常有“最新、版本、活动、赛季、更新、公告、上线、调整”等词。

```text
最新版本更新了什么
这个赛季有什么活动
王者荣耀今天有什么公告
新英雄什么时候上线
最近哪些英雄被削弱了
S36 赛季改动有哪些
周年庆活动有什么奖励
体验服最近更新了什么
本周限免英雄有哪些
最近皮肤返场消息
```

### 攻略类 guide

这类 Query 通常是在问“怎么玩、怎么打、怎么出装、连招、打法、技巧、阵容”。

```text
李白怎么玩
韩信打野怎么刷野
孙尚香怎么出装
后羿怎么打团
貂蝉连招顺序是什么
新手适合玩什么英雄
打野前期怎么带节奏
射手怎么防刺客
王者荣耀怎么上分
吕布对线怎么打
```

------

# 四、BERT 微调模型设计

## 1. 模型结构

使用 BERT 做三分类：

```text
输入 Query
   ↓
Tokenizer 分词
   ↓
BERT 编码
   ↓
取 [CLS] 向量
   ↓
全连接分类层
   ↓
Softmax 输出三类概率
```

数学形式可以写成：

```text
h_cls = BERT(query)[CLS]
p = softmax(W h_cls + b)
```

其中：

```text
p_document = 资料类概率
p_news = 资讯类概率
p_guide = 攻略类概率
```

训练目标使用交叉熵损失：

```text
L = - Σ y_i log(p_i)
```

------

## 2. 推荐模型

中文 RAG 系统可以选择：

| 模型                        | 说明                       | 推荐程度 |
| --------------------------- | -------------------------- | -------- |
| bert-base-chinese           | 官方中文 BERT，稳定        | 推荐     |
| hfl/chinese-roberta-wwm-ext | 中文效果通常比原始 BERT 好 | 很推荐   |
| hfl/chinese-macbert-base    | 中文语义理解较好           | 很推荐   |
| TinyBERT                    | 推理快，适合低延迟         | 部署推荐 |
| DistilBERT                  | 速度较快                   | 可选     |

如果只是做三分类，推荐优先选择：

```text
hfl/chinese-macbert-base
```

或者：

```text
hfl/chinese-roberta-wwm-ext
```

如果部署机器性能一般，可以选择：

```text
uer/chinese_roberta_L-4_H-512
TinyBERT
```

------

# 五、训练参数建议

对于三分类任务，参数可以这样设置：

```text
max_length = 64
batch_size = 16 或 32
learning_rate = 2e-5
epochs = 3 到 5
weight_decay = 0.01
warmup_ratio = 0.1
dropout = 0.1
optimizer = AdamW
loss = CrossEntropyLoss
```

为什么 `max_length = 64` 就够？

因为意图识别的输入通常很短，例如：

```text
李白怎么玩
最新版本更新了什么
孙尚香技能介绍
```

大多数 Query 不超过 30 个中文字符，所以没必要设置成 128 或 256。长度越短，推理越快。

------

# 六、训练代码示例

下面是一个基于 Hugging Face Transformers 的核心训练示例。

## 1. 安装依赖

```bash
pip install torch transformers datasets scikit-learn pandas
```

------

## 2. 数据格式

假设数据文件是：

```text
data/intent_train.csv
data/intent_valid.csv
```

内容如下：

```csv
query,label
孙尚香一技能是什么,document
最新版本更新了什么,news
李白怎么玩,guide
```

------

## 3. 训练代码

文件名：

```text
train_intent_bert.py
```

代码：

```python
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "hfl/chinese-macbert-base"

label2id = {
    "document": 0,
    "news": 1,
    "guide": 2
}

id2label = {
    0: "document",
    1: "news",
    2: "guide"
}


def load_csv(path):
    df = pd.read_csv(path)
    df["label"] = df["label"].map(label2id)
    return Dataset.from_pandas(df)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(example):
    return tokenizer(
        example["query"],
        truncation=True,
        padding="max_length",
        max_length=64
    )


train_dataset = load_csv("data/intent_train.csv")
valid_dataset = load_csv("data/intent_valid.csv")

train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["query"])
valid_dataset = valid_dataset.remove_columns(["query"])

train_dataset.set_format("torch")
valid_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro"
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1
    }


training_args = TrainingArguments(
    output_dir="./intent_bert_output",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("./intent_bert_model")
tokenizer.save_pretrained("./intent_bert_model")
```

------

# 七、推理代码示例

文件名：

```text
predict_intent.py
```

代码：

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./intent_bert_model"

id2label = {
    0: "document",
    1: "news",
    2: "guide"
}

label_name = {
    "document": "资料类",
    "news": "资讯类",
    "guide": "攻略类"
}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_intent(query: str):
    inputs = tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    intent = id2label[pred_id]
    confidence = float(probs[pred_id].item())

    return {
        "query": query,
        "intent": intent,
        "intent_name": label_name[intent],
        "confidence": confidence
    }


if __name__ == "__main__":
    test_queries = [
        "孙尚香一技能是什么",
        "最新版本更新了什么",
        "李白怎么玩"
    ]

    for q in test_queries:
        print(predict_intent(q))
```

输出示例：

```json
{
  "query": "李白怎么玩",
  "intent": "guide",
  "intent_name": "攻略类",
  "confidence": 0.9821
}
```

------

# 八、接入 RAG 系统的方式

## 1. 意图识别服务返回结果

推荐封装成一个独立服务：

```text
POST /intent/predict
```

请求：

```json
{
  "query": "李白怎么玩"
}
```

返回：

```json
{
  "intent": "guide",
  "intent_name": "攻略类",
  "confidence": 0.9821
}
```

------

## 2. Java RAG 系统中的调用位置

可以放在：

```text
QueryUnderstandingService
```

例如：

```java
public QueryUnderstandingResult understand(String query) {
    IntentResult intentResult = intentClassifierClient.predict(query);

    List<String> rewrittenQueries = queryRewriteService.rewrite(query, intentResult);

    ExtractedInfo extractedInfo = informationExtractor.extract(query, intentResult);

    RetrievalRoute route = retrievalRouter.route(intentResult, extractedInfo);

    return new QueryUnderstandingResult(
        query,
        intentResult,
        rewrittenQueries,
        extractedInfo,
        route
    );
}
```

------

# 九、根据意图选择不同检索策略

这是 BERT 意图识别真正有价值的地方。

## 1. 资料类：document

适合查静态知识。

例如：

```text
孙尚香一技能是什么
破军属性是什么
红 buff 作用是什么
```

检索策略：

```text
1. 优先检索静态知识库
2. 向量召回 topK = 20
3. 关键词召回 topK = 20
4. RRF 融合
5. Rerank 取 topK = 5
6. LLM 严格基于知识库回答
```

适合的 Prompt：

```text
请根据以下资料回答用户问题。
如果资料中没有答案，请回答“知识库中暂未找到相关资料”，不要编造。
```

------

## 2. 资讯类：news

适合查公告、版本、赛季、活动。

例如：

```text
最新版本更新了什么
这个赛季有什么活动
最近哪些英雄调整了
```

检索策略：

```text
1. 优先检索资讯库、公告库、版本更新库
2. 必须加入时间字段过滤
3. 优先返回最新内容
4. 向量召回 topK = 30
5. 关键词召回 topK = 30
6. 按发布时间加权
7. Rerank 取 topK = 5
```

资讯类建议在知识库中增加字段：

```text
publish_time
source
version
season
event_type
```

排序时可以加入时间权重：

```text
final_score = 0.7 * rerank_score + 0.3 * time_score
```

适合的 Prompt：

```text
请根据最新资讯内容回答用户问题。
回答时需要说明信息来源时间。
如果没有找到最新资讯，请明确说明当前知识库没有相关最新信息。
```

------

## 3. 攻略类：guide

适合查玩法、出装、连招、对线、阵容。

例如：

```text
李白怎么玩
孙尚香怎么出装
打野怎么带节奏
```

检索策略：

```text
1. 优先检索攻略库、玩法库、英雄技巧库
2. Query 改写时扩展玩法相关词
3. 向量召回 topK = 40
4. 关键词召回 topK = 20
5. Rerank 取 topK = 8
6. LLM 输出结构化步骤
```

攻略类可以扩展 Query：

```text
原始 Query：
李白怎么玩

扩展 Query：
李白 技能连招
李白 打野思路
李白 出装铭文
李白 团战技巧
李白 新手攻略
```

适合的 Prompt：

```text
请根据攻略资料，按照“技能理解、出装推荐、连招思路、对线/打野技巧、团战打法、注意事项”的结构回答。
```

------

# 十、低置信度兜底机制

不要完全相信分类结果。建议设置置信度阈值。

```text
confidence >= 0.75：直接使用预测意图
0.50 <= confidence < 0.75：使用预测意图，但同时扩大召回范围
confidence < 0.50：进入 UNKNOWN，走通用混合检索
```

示例：

```python
if confidence < 0.5:
    intent = "unknown"
```

Java 中可以这样处理：

```java
if (intentResult.getConfidence() < 0.5) {
    route = RetrievalRoute.GENERAL_HYBRID;
} else if (intentResult.getIntent().equals("news")) {
    route = RetrievalRoute.NEWS_FIRST;
} else if (intentResult.getIntent().equals("guide")) {
    route = RetrievalRoute.GUIDE_FIRST;
} else {
    route = RetrievalRoute.DOCUMENT_FIRST;
}
```

------

# 十一、容易混淆的样本处理

有些 Query 会比较模糊。

## 1. “李白介绍”

可能是资料类，也可能是攻略类。

建议标注为：

```text
资料类 document
```

因为“介绍”更偏静态资料。

------

## 2. “李白怎么玩”

标注为：

```text
攻略类 guide
```

因为“怎么玩”明显是玩法攻略。

------

## 3. “李白最近改了吗”

标注为：

```text
资讯类 news
```

因为“最近”“改了”表示时效信息。

------

## 4. “孙尚香强不强”

这个比较模糊，可能是攻略类，也可能是资讯类。

建议根据系统目标处理：

```text
如果问当前版本强度 → 资讯类
如果问玩法强度分析 → 攻略类
```

可以扩展为四类：

```text
资料类 document
资讯类 news
攻略类 guide
综合类 general
```

但你现在只需要三类的话，可以先归到：

```text
攻略类 guide
```

------

# 十二、单次识别大概多长时间？

这个和模型大小、硬件、部署方式有关。

以 **BERT-base / MacBERT-base，max_length=64，batch_size=1** 为例：

| 部署环境                     | 单条 Query 识别时间        |
| ---------------------------- | -------------------------- |
| 普通 CPU，PyTorch 原生推理   | 30 ms ~ 150 ms             |
| 较好的 CPU + ONNX Runtime    | 10 ms ~ 50 ms              |
| CPU + INT8 量化              | 5 ms ~ 30 ms               |
| 普通 GPU，例如 RTX 3060 / T4 | 3 ms ~ 15 ms               |
| TinyBERT / 小模型 CPU 推理   | 3 ms ~ 20 ms               |
| 远程 HTTP 调用模型服务       | 20 ms ~ 200 ms，取决于网络 |

比较实际的估计：

```text
本地 CPU 部署 BERT-base：单次大约 50 ms 左右
本地 GPU 部署 BERT-base：单次大约 5 到 15 ms
ONNX + CPU 优化后：单次大约 10 到 30 ms
TinyBERT CPU 部署：单次大约 5 到 15 ms
```

如果你的 RAG 系统本身还要做：

```text
Query 改写
Milvus 检索
Rerank
LLM 生成
```

那么 BERT 意图识别的耗时通常不是瓶颈。

整体耗时大概是：

| 模块           | 典型耗时       |
| -------------- | -------------- |
| BERT 意图识别  | 5 ms ~ 100 ms  |
| Embedding 生成 | 20 ms ~ 200 ms |
| Milvus 检索    | 10 ms ~ 100 ms |
| Rerank         | 50 ms ~ 500 ms |
| LLM 生成       | 1 s ~ 10 s     |

所以，意图识别一般只占很小一部分。

------

# 十三、推荐部署方案

## 方案一：Python 模型服务 + Java 调用

这是最简单的方式。

```text
SpringBoot RAG 系统
        ↓ HTTP
Python FastAPI 意图识别服务
        ↓
BERT 模型
```

优点：

```text
1. 训练和推理都方便
2. Hugging Face 支持好
3. 修改模型不影响 Java 主项目
```

缺点：

```text
需要额外部署一个 Python 服务
```

------

## 方案二：导出 ONNX，Java 本地推理

```text
BERT 模型
   ↓
导出 ONNX
   ↓
Java 使用 ONNX Runtime 推理
   ↓
SpringBoot 内部直接调用
```

优点：

```text
1. 不需要 Python 服务
2. 延迟低
3. 更适合生产部署
```

缺点：

```text
1. ONNX 导出和 tokenizer 处理稍微麻烦
2. Java 中实现分词需要额外处理
```

------

## 方案三：使用小模型替代 BERT-base

如果只是三分类，可以使用：

```text
TinyBERT
MiniLM
RoBERTa-small
```

优点：

```text
速度更快
CPU 也能轻松部署
```

缺点：

```text
准确率可能略低
```

但对于“资料类 / 资讯类 / 攻略类”这种意图分类任务，小模型通常已经够用。

------

# 十四、推荐项目结构

```text
intent-classifier/
│
├── data/
│   ├── intent_train.csv
│   ├── intent_valid.csv
│   └── intent_test.csv
│
├── scripts/
│   ├── build_dataset.py
│   ├── train_intent_bert.py
│   ├── predict_intent.py
│   └── export_onnx.py
│
├── model/
│   └── intent_bert_model/
│
├── service/
│   ├── app.py
│   └── intent_service.py
│
├── requirements.txt
└── README.md
```

如果接入 Java RAG 项目，可以在 Java 项目中新增：

```text
src/main/java/com/example/gamerag/intent/
│
├── IntentResult.java
├── IntentType.java
├── IntentClassifierClient.java
└── IntentRouteService.java
```

------

# 十五、RAG 中的最终路由逻辑

可以设计成：

```text
资料类 document：
    检索 hero_knowledge、skill_knowledge、item_knowledge、mechanism_knowledge

资讯类 news：
    检索 update_news、season_news、event_news、announcement_news
    增加 publish_time 排序

攻略类 guide：
    检索 hero_guide、build_guide、combo_guide、teamfight_guide
    增加 Query 扩展
```

伪代码：

```java
public RetrievalConfig buildRetrievalConfig(IntentResult intentResult) {
    switch (intentResult.getIntent()) {
        case "document":
            return RetrievalConfig.builder()
                    .collections(List.of("hero_knowledge", "item_knowledge", "skill_knowledge"))
                    .denseTopK(20)
                    .sparseTopK(20)
                    .rerankTopK(5)
                    .timeSensitive(false)
                    .build();

        case "news":
            return RetrievalConfig.builder()
                    .collections(List.of("update_news", "event_news", "announcement_news"))
                    .denseTopK(30)
                    .sparseTopK(30)
                    .rerankTopK(5)
                    .timeSensitive(true)
                    .sortByTime(true)
                    .build();

        case "guide":
            return RetrievalConfig.builder()
                    .collections(List.of("hero_guide", "build_guide", "combo_guide"))
                    .denseTopK(40)
                    .sparseTopK(20)
                    .rerankTopK(8)
                    .enableQueryExpansion(true)
                    .build();

        default:
            return RetrievalConfig.defaultHybrid();
    }
}
```

------

# 十六、完整落地流程

建议按这个顺序做：

```text
第一步：确定意图标签
document / news / guide

第二步：整理训练数据
每类至少 1000 条 Query

第三步：训练 BERT 三分类模型
使用 chinese-macbert-base 或 chinese-roberta-wwm-ext

第四步：评估模型
重点看 macro-F1、混淆矩阵、每类召回率

第五步：加入低置信度兜底
confidence < 0.5 时走通用检索

第六步：封装推理服务
FastAPI 或 ONNX Runtime

第七步：接入 Java RAG 系统
放在 QueryUnderstandingService 前端

第八步：根据意图配置不同检索策略
资料类、资讯类、攻略类走不同 collection、topK、排序方式

第九步：线上日志回流
把分类错误的 Query 收集起来继续训练
```

------

# 十七、最终建议

你的场景里，BERT 意图识别不是最复杂的部分，但它对 RAG 检索质量很关键。

推荐初版采用：

```text
模型：hfl/chinese-macbert-base
类别：资料类 / 资讯类 / 攻略类
max_length：64
训练数据：每类 1000 条起步
部署：Python FastAPI 服务
单次耗时：CPU 约 30~100 ms，GPU 约 5~15 ms
```

如果后期要上线并降低延迟，再优化为：

```text
MacBERT-base → TinyBERT / MiniLM
PyTorch → ONNX Runtime
FP32 → INT8 量化
Python 服务 → Java 本地 ONNX 推理
```

最终在 RAG 中的作用是：

```text
不是直接回答问题，
而是判断用户到底想查资料、查资讯，还是查攻略，
然后决定后续怎么改写 Query、查哪个知识库、召回多少内容、如何排序。
```