## Summary: Evaluation Metrics (Binary Labels, Top-25 Retrieval)

### 1️⃣ Keep K Fixed

* Do **not** change K per query based on how many relevant items exist.
* Fixed K ensures:

  * Fair comparison across queries
  * Fair comparison across model versions
  * Reproducibility
>NOTE: Most of the time k=response_limit aka the number of results returned by the VectorDBAdapter.search() method.
---

### 2️⃣ Problem Context

* Labels are **binary (relevant / non-relevant)**.
* The **average number of relevant items per query varies across benchmarks**.
* Relevant items are often sparse compared to total corpus size.
* Early ranking quality matters more than full-corpus coverage.

---

### 3️⃣ Recommended Metric Strategy

Use a combination of ranking-sensitive and top-K metrics:

### Primary Decision Metrics

* **MRR (Mean Reciprocal Rank)**

  * Measures how early the first relevant result appears.
  * Very stable when relevant counts are small.
  * Strong indicator of ranking quality.

* **Success@k (Hit Rate@k)**

  * Measures whether at least one relevant result appears in the top k.
  * Easy to interpret for stakeholders.
  * Reflects practical user satisfaction.

* **Diversity (1 − ILS)**

  * Measures how diverse (non-redundant) the top-k list is based on the cosine similarity of the retrieved item vectors. Meaning if the vector was computed using the image and caption, the diversity would depend on the image and caption. This means that metadata (eg; location, time, etc) would not be considered for diversity. Unless you are also computing the vector using the metadata.
  * ILS (Intra-List Similarity) is the average pairwise cosine similarity of the retrieved item vectors; diversity = 1 − ILS, so **higher is more diverse**.
  * Requires result vectors (e.g. returned by Weaviate/Milvus adapters under the `"vector"` column). When vectors are missing or invalid, diversity is reported as NaN.
>NOTE:If diversity is not important: For example, if you only care that the top results include relevant items, but not necessarily diverse. Use this as a Supporting Metric.
---

### Supporting Metrics

* **Precision@k**

  * Measures how clean the first page of results is.
  * Directly aligned with user-visible output.

* **NDCG@k**

  * Evaluates ranking quality with position discounting.
  * Useful even with binary labels.

* **Recall@k**

  * Measures how much of the relevant set appears in the top k.
  * Interpreted carefully when relevant counts vary across benchmarks.

---

### 4️⃣ Interpretation Guidance

* Precision reflects first-page quality.
* MRR reflects early ranking strength.
* Success@k reflects practical usefulness.
* NDCG captures ranking structure.
* Metric comparisons should primarily be made:

  * Within the same benchmark
  * Across model versions

Cross-benchmark comparisons require awareness of differing relevant densities.

---

### 5️⃣ Standard Reporting Recommendation

For each benchmark, report:

* MRR
* Success@k
* Precision@k
* NDCG@k
* Recall@k

Use MRR and Precision@k as primary model selection signals.
