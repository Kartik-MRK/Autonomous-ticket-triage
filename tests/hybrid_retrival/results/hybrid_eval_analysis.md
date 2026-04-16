# 🧑‍⚖️ Hybrid Retrieval Evaluation — Judge's Analysis

> **Evaluation Date:** 2026-04-14  
> **Mode:** Hybrid (Dense + BM25 + RRF) with HyDE  
> **Queries Evaluated:** 100 test issues (`test_processed.json`)  
> **Retrieval Config:** `final_top_k=60` retrieved, reranked → `top_k=5` evaluated  
> **Total Runtime:** ~53 minutes (3185.6 seconds)

---

## 📊 Raw Results at a Glance

```json
{
  "retrieval_mode": "hybrid",
  "hyde_enabled": true,
  "routing_classification": {
    "top1_accuracy":  0.02,
    "top3_accuracy":  0.81,
    "top5_accuracy":  0.89,
    "precision":      0.000976,
    "recall":         0.02,
    "f1_score":       0.001860
  },
  "retrieval_rag": {
    "k":              5,
    "precision_at_k": 0.528,
    "recall_at_k":    0.0307,
    "mrr":            0.7006,
    "ndcg_at_k":      0.5342
  },
  "meta": {
    "num_queries":    100,
    "hyde_triggers":  2,
    "elapsed_seconds": 3185.6
  }
}
```

---

## 📖 What Does Each Metric Mean?

### Section 1 — `routing_classification`

This section measures how well the **LLM classifier** (`classify_ticket` via Gemini) correctly routes a ticket to the right team. It compares the **predicted team label** against the **ground-truth team** from the dataset.

| Metric | Definition | Formula |
|--------|-----------|---------|
| **`top1_accuracy`** | Fraction of queries where the classifier's single predicted team **exactly** matched the ground-truth team | `correct_top1 / total` |
| **`top3_accuracy`** | Fraction of queries where the ground-truth team appeared **within the top-3 teams** of retrieved documents OR the classifier was correct | `(top3_hits) / total` |
| **`top5_accuracy`** | Same as top3, but expanded to the **top-5 retrieved documents**' teams | `(top5_hits) / total` |
| **`precision`** | Weighted average precision across all teams — "of all tickets classified as team X, what fraction truly belonged to X?" Uses `sklearn weighted average` | `sklearn.precision_score(y_true, y_pred, average='weighted')` |
| **`recall`** | Weighted average recall across all teams — "of all tickets that truly belonged to team X, what fraction did we catch?" | `sklearn.recall_score(y_true, y_pred, average='weighted')` |
| **`f1_score`** | Harmonic mean of precision and recall. A single combined quality score for classification | `2 × (P × R) / (P + R)` |

> **Key insight:** `top1_accuracy` only rewards the exact predicted label. `top3/top5_accuracy` is more lenient — it checks whether the **retrieved context** contained the correct team, which is more reflective of how the retrieval aids the pipeline.

---

### Section 2 — `retrieval_rag`

This section measures how well the **retrieval system** (hybrid BM25 + dense + RRF + reranker) returns documents relevant to the query. Relevance is defined as: **a retrieved document belongs to the same team as the query's ground-truth team**.

| Metric | Definition | Formula |
|--------|-----------|---------|
| **`k`** | Number of documents evaluated (top-K slice used for scoring) | Fixed at `5` |
| **`precision_at_k`** | Among the top-K retrieved documents, what fraction are relevant? | `relevant_in_top_k / k` |
| **`recall_at_k`** | Among ALL relevant documents in the corpus, what fraction did we retrieve in the top-K? | `relevant_in_top_k / total_relevant_in_corpus` |
| **`mrr`** | Mean Reciprocal Rank — "how quickly does the first relevant document appear?" A score of 1.0 means it always appeared first; 0.5 means it appeared on average at rank 2 | `mean(1 / rank_of_first_relevant)` |
| **`ndcg_at_k`** | Normalized Discounted Cumulative Gain — rewards relevant results appearing as early in the list as possible, normalized against the ideal ranking | `DCG / IDCG` |

> **Key insight:** `recall_at_k` is measured against the **full corpus** size for that team, not against a fixed "5 relevant docs" pool. When a team has hundreds of issues in the corpus, retrieving only 1–2 in k=5 yields a tiny recall even if those 1–2 are perfectly relevant.

---

## 🧑‍⚖️ Judge's Verdict — Detailed Analysis

### 1. Routing Classification — Mixed Picture

```
top1_accuracy  = 0.02  ← 🔴 Critically low
top3_accuracy  = 0.81  ← 🟢 Good
top5_accuracy  = 0.89  ← 🟢 Very good
precision      = 0.00098 ← 🔴 Near-zero
recall         = 0.02  ← 🔴 Near-zero
f1_score       = 0.00186 ← 🔴 Near-zero
```

**Top-1 Accuracy (0.02 / 2%):** This is the most alarming number in the report. Out of 100 test queries, the classifier correctly predicted the exact team label in **only 2 cases**. Browsing the `query_responses.json` reveals the pattern clearly — the classifier consistently predicts coarse labels like `"backend"` or `"frontend"` when the actual ground truth is fine-grained team names like `"css parsing and computation"`, `"html parser"`, `"selection"`, `"core & html"` etc. The spaces are completely mismatched.

**Top-3/5 Accuracy (0.81 / 0.89):** These numbers are actually very healthy. They say that in 89% of queries, the correct team **did appear** somewhere in the top-5 retrieved documents. This means the **retrieval pipeline is doing its job well** — it is surfacing the right documents. The failure lies entirely in classification, not retrieval.

**Precision / Recall / F1 (near-zero):** These are `sklearn` multi-class weighted metrics comparing exact string labels. Since the classifier predicts "backend" / "frontend" / "devops" (generic buckets) while ground truth is "html parser" / "tables" / "networking" (specific team names), there is essentially **zero label overlap**. The classifier is operating in a completely different label space than the evaluation ground truth. This is a **label mismatch problem**, not a capability problem.

---

### 2. Retrieval RAG — Solid Core Performance

```
precision_at_k = 0.528  ← 🟡 Decent (above random)
recall_at_k    = 0.031  ← 🔴 Very low (structural reason)
mrr            = 0.701  ← 🟢 Strong
ndcg_at_k      = 0.534  ← 🟡 Reasonable
```

**Precision@5 (0.528):** On average, roughly **2.6 out of 5 retrieved documents are relevant**. This is actually quite solid for a zero-shot retrieval system on a multi-team bug corpus. It means more than half the retrieved context is on-target. Some queries (like `id=960`, `id=961`, `id=1039`) achieve a perfect **precision@5 = 1.0**, while others (like `id=1059` — "floats within tables") score only **0.2** because the retriever confuses layout issues with table issues.

**Recall@5 (0.031):** This number looks terrible, but it is **structurally unavoidable**. Here is why:
- The corpus likely contains hundreds of documents per team (e.g., "HTML Parser" might have 200+ issues).
- We only retrieve top-5 documents.
- `recall_at_k = relevant_found_in_k / total_relevant_in_corpus`.
- So even if all 5 retrieved documents are relevant, if the corpus has 200 relevant documents, recall = 5/200 = **0.025**.
- This is a corpus-size artifact, **not a retrieval failure**.

**MRR (0.701):** This is the standout number. An MRR of 0.70 means the first relevant document appears, on average, within the **top 1.4 positions**. In practice, looking at individual responses: queries with homogeneous teams (like CSS, HTML Parser, UI Events) reliably have a relevant document at rank 1. The hybrid retrieval is very good at putting an on-target document near the top.

**NDCG@5 (0.534):** This accounts for the **ordering** of results. A score of 0.534 (out of 1.0) means the retrieved ranking is moderately close to the ideal ranking. It is penalized when relevant docs appear at positions 3–5 rather than 1–2. The variance across queries is high (0.169 to 1.0), which suggests the system is strong on "easy" well-separated topics but struggles with overlapping ones (e.g., Layout vs. Tables vs. CSS).

---

## 🔴 Why Is Classification Precision So Low?

This is the central question, and the answer is **label space mismatch**.

### Root Cause Analysis

| Problem | Description |
|---------|-------------|
| **Generic classifier output** | The `classify_ticket` function returns broad organisational labels like `"backend"`, `"frontend"`, or `"devops"` |
| **Fine-grained ground truth** | The test dataset uses specific team names: `"css parsing and computation"`, `"html parser"`, `"dom: selection"`, `"tables"`, `"networking"`, etc. |
| **Zero overlap** | `sklearn.precision_score` compares these as exact strings — `"backend" ≠ "html parser"` — so effectively every prediction fails |
| **HyDE barely triggered** | HyDE only fired 2/100 times, so it did not affect classification at all |

### Evidence from `query_responses.json`

Looking at several examples:

| Issue | True Team | Predicted Team | Top1 Correct |
|-------|-----------|---------------|--------------|
| 573 (link stylesheet) | `core & html` | `backend` | ❌ |
| 758 (text crash) | `selection` | `frontend` | ❌ |
| 815 (noscript tag) | `html parser` | `backend` | ❌ |
| 960 (window.onresize) | `ui events & focus handling` | `frontend` | ❌ |
| 1039 (negative font-size) | `css parsing and computation` | `backend` | ❌ |
| 1044 (table inheritance) | `css parsing and computation` | `frontend` | ❌ |

The classifier never predicts domain-specific team names. It maps everything to broad categorical buckets. The precision/recall/f1 are near-zero because `sklearn` treats this as a multi-class problem where **no predicted class matches the true class in 98% of cases**.

---

## 🛠️ How to Fix This — Actionable Improvements

### Fix 1: Force Classifier to Use the Known Team Label Space

> **Priority: HIGH | Impact: HIGH**

The classifier should be given the list of actual valid team labels and forced to pick from them.

```python
# In modules/classifier.py or classify_ticket()
# Pass known_teams into the classification prompt

TEAM_LABELS = [
    "html parser", "css parsing and computation", "tables",
    "layout", "networking", "selection", "core & html",
    "editor", "ui events & focus handling", ...
]

prompt = f"""
You are a bug triage assistant. Classify the following ticket.
You MUST assign the ticket to exactly one of these teams:
{TEAM_LABELS}

Ticket:
{description}

Return JSON: {{ "type": "...", "severity": "...", "team": "<one of the above teams>" }}
"""
```

This single change would convert the classification metrics from near-zero to realistic values. The model knows the domain — it just lacks constraints.

---

### Fix 2: Use Retrieval-Augmented Classification (RAC)

> **Priority: HIGH | Impact: HIGH**

Instead of asking the classifier to guess the team from the ticket alone, **inject the top retrieved teams** as candidate labels.

```python
retrieved_teams = list({doc["metadata"]["team"] for doc in reranked_docs[:5]})

prompt = f"""
Based on the following similar resolved tickets, the most likely teams are:
{retrieved_teams}

Classify this ticket into one of these teams or the closest one.
Ticket: {description}
"""
```

The retrieval already surfaces the correct team in 89% of cases (top-5). Using those retrieved team names as the classification prompt would dramatically boost top-1 accuracy.

---

### Fix 3: Use a Lightweight Multi-Class Classifier on Top of Embeddings

> **Priority: MEDIUM | Impact: HIGH**

Train a simple classifier (logistic regression or lightweight MLP) on top of the embedding vectors to predict the fine-grained team label. The dense embeddings already contain rich semantic information. A small classifier trained on the corpus labels would:
- Be very fast (no LLM call needed for routing)
- Be directly calibrated to the actual team label space
- Achieve much higher precision naturally

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Train phase (offline, using train_processed.json embeddings)
clf = LogisticRegression(max_iter=1000)
clf.fit(embedding_matrix, team_labels)

# At inference time
predicted_team = clf.predict([query_embedding])[0]
```

---

### Fix 4: Post-Process Classifier Output with Team Mapping

> **Priority: LOW | Impact: MEDIUM**

If changing the classifier prompt is not immediately feasible, add a **post-processing mapping layer**:

```python
GENERIC_TO_TEAMS = {
    "backend": ["html parser", "networking", "javascript engine"],
    "frontend": ["css parsing and computation", "layout", "tables", "selection"],
    "infrastructure": ["build system", "testing"],
}

def refine_team(predicted: str, retrieved_teams: list) -> str:
    candidates = GENERIC_TO_TEAMS.get(predicted, [])
    for t in retrieved_teams:
        if t in candidates:
            return t  # Use the retrieved team that matches the predicted bucket
    return retrieved_teams[0] if retrieved_teams else predicted
```

---

### Fix 5: Improve Retrieval Recall (Increase k)

> **Priority: MEDIUM | Impact: MEDIUM for RAG**

Recall@k will always be low when k=5 and the corpus is large. For a RAG use case, consider:
- Increasing `k` to 10 or 20 for the retrieval evaluation — this gives the system more chances to surface relevant documents
- Using **per-query recall normalization** (capping `total_relevant` at min(corpus_count, 50)) to avoid artificially tiny recall scores on large teams
- Reporting **R-Precision** (precision at the number of relevant documents) instead of Recall@5 for a fairer metric

---

### Fix 6: Improve HyDE Trigger Rate

> **Priority: LOW | Impact: LOW (small gain)**

HyDE triggered only 2/100 times. The confidence threshold in `settings.HYDE_CONFIDENCE_THRESHOLD` may be too aggressive (too low a threshold, so most queries "pass" without triggering). Review:
- Lower the threshold slightly so HyDE activates on more ambiguous queries (e.g., those with MRR < 0.5)
- Alternatively, trigger HyDE based on **prediction confidence from the classifier**, not retrieval score

---

## 📈 Summary Scorecard

| Area | Score | Status |
|------|-------|--------|
| Top-1 Classification Accuracy | 2% | 🔴 Critical — label mismatch |
| Top-3 Retrieval Coverage | 81% | 🟢 Good |
| Top-5 Retrieval Coverage | 89% | 🟢 Very Good |
| Classifier Label Precision | ~0.1% | 🔴 Critical — label mismatch |
| Retrieval Precision@5 | 52.8% | 🟡 Acceptable |
| Retrieval Recall@5 | 3.1% | 🔴 Structural (corpus size) |
| MRR | 70.1% | 🟢 Strong |
| NDCG@5 | 53.4% | 🟡 Reasonable |
| HyDE Utility | 2% trigger rate | 🟡 Rarely needed |

---

## 🎯 Priority Fix Roadmap

```
Priority 1 (Fix Now):
└── Constrain classifier to use known team label space (Fix 1)
    → Expected impact: top1_accuracy 0.02 → ~0.50+
    → Expected impact: precision 0.001 → ~0.45+

Priority 2 (Next Sprint):
└── Use retrieved teams as classification candidates (Fix 2)
    → Expected impact: top1_accuracy → 0.70+ (leverages existing 89% top-5 coverage)

Priority 3 (Research):
└── Train lightweight embedding classifier (Fix 3)
    → Fastest inference, highest accuracy ceiling for routing

Priority 4 (Metrics):
└── Fix recall_at_k computation to use capped total_relevant (Fix 5)
    → Makes recall a meaningful, comparable metric
```

---

## 🔍 Observations from Individual Queries

- **Best performing queries:** Issues with highly specific, homogeneous vocabulary (e.g., `id=1039` — "negative CSS font-size" → all 5 retrieved docs from CSS team → P@5 = 1.0, MRR = 1.0, NDCG = 1.0).
- **Worst performing queries:** Cross-cutting layout issues (e.g., `id=1059` — "floats within tables" → retriever confused Layout team with Tables team → P@5 = 0.2, NDCG = 0.17).
- **HyDE case (`id=1124`):** HyDE was activated for the entity-handling bug. The query had very low cosine similarity (top score = 0.149), suggesting the query was very different from indexed docs. HyDE improved retrieval but still correctly surfaced HTML Parser results.
- **The 2 top-1 correct predictions** were almost certainly cases where the ticket description explicitly mentioned a team or component name that coincidentally matched what the classifier predicted as a broad bucket.

---

*Analysis generated on 2026-04-14 | Evaluation Engine: `tests/eval_engine.py` | Results: `tests/hybrid_retrival/results/evaluation_metrics.json`*
