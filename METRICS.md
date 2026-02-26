## Summary: Evaluation Metrics for Sage Image Search (Binary Labels, Top-25 Retrieval)

### 1️⃣ Keep K Fixed

* Do **not** change K per query based on how many relevant items exist.
* Fixed K ensures:

  * Fair comparison across queries
  * Fair comparison across model versions
  * Reproducibility

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

  * Measures whether at least one relevant result appears in the top 25.
  * Easy to interpret for stakeholders.
  * Reflects practical user satisfaction.

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
