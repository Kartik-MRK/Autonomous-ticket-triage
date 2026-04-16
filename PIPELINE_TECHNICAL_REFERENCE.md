# Autonomous Ticket Triage — Deep Technical Reference

> **Purpose:** Low-level walkthrough of every stage in the pipeline, every model used, every threshold configured, how the prompt engineering works, and exactly how each evaluation metric is computed.

---

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Stage 0 — Data Ingestion & Preprocessing](#2-stage-0--data-ingestion--preprocessing)
3. [Stage 1 — Embedding & Indexing](#3-stage-1--embedding--indexing)
4. [Stage 2 — Hybrid Retrieval (Dense + Sparse + Weighted RRF)](#4-stage-2--hybrid-retrieval)
5. [Stage 3 — Cross-Encoder Reranking](#5-stage-3--cross-encoder-reranking)
6. [Stage 4 — Retrieval-Augmented Classification (RAC)](#6-stage-4--retrieval-augmented-classification)
7. [Stage 4b — HyDE Fallback](#7-stage-4b--hyde-fallback)
8. [Stage 5 — RAG Response Generation](#8-stage-5--rag-response-generation)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. System Architecture Overview

```
Incoming Ticket (title + description + labels)
        │
        ▼
┌────────────────────┐
│  Stage 0           │  regex clean + spaCy lemmatize
│  Preprocessing     │  → unified_text, tokens, entities
└────────┬───────────┘
         │
         ▼
┌────────────────────┐     ┌──────────────────────────┐
│  Stage 1           │     │  Offline (build-index)   │
│  Embedding         │────▶│  ChromaDB (dense vectors)│
│  BAAI/bge-large-en │     │  BM25 in-memory index    │
└────────┬───────────┘     └──────────────────────────┘
         │
         ▼
┌────────────────────┐
│  Stage 2           │  Dense top-30 + Sparse top-30
│  Hybrid Retrieval  │  → Weighted RRF → top-60 candidates
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Stage 3           │  BAAI/bge-reranker-large
│  Cross-Encoder     │  → top-10 reranked docs with scores
│  Reranking         │
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Stage 4           │  Ollama (llama3.1:8b)
│  Classification    │  known_teams + retrieved context
│  (RAC)             │  → {type, severity, team}
└────────┬───────────┘
         │
         ▼
┌────────────────────┐     fires if agreement(top-10) < 0.3
│  Stage 4b          │────▶ generate hypothetical doc → re-retrieve
│  HyDE Fallback     │      → re-rerank → re-classify
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Stage 5           │  Ollama (llama3.1:8b)
│  RAG Generation    │  top-5 retrieved docs as context
│                    │  → debugging suggestions + fix directions
└────────────────────┘
```

---

## 2. Stage 0 — Data Ingestion & Preprocessing

**Module:** `modules/preprocessing.py`  
**Triggered by:** `python main.py build-index`

### What happens

Every raw GitHub/Bugzilla issue goes through two cleaning passes:

#### Pass 1 — Regex Cleaning (`clean_text_regex`)
Applied patterns in order:
| Pattern | What it removes |
|---------|----------------|
| ` ``` ... ``` ` | Fenced code blocks (replaced with space) |
| `` `code` `` | Inline code spans |
| `https?://...` | URLs |
| `<tag>` | HTML tags |
| `![text](url)` | Markdown image/link syntax (keeps link text) |
| `^#{1,6}` | Markdown headers |
| `^---` / `===` | Horizontal rules |
| `^- ` / `^* ` | Bullet list markers |
| `@username` | Mentions |
| `#abc123` | Hex color codes |
| `/path/to/file` | File paths (2+ segments) |
| `[^\w\s.,;:!?"-]` | Remaining special characters |
| `\s+` | Excessive whitespace |

#### Pass 2 — spaCy NLP (`process_with_spacy`)
Model: `en_core_web_sm`

- Removes stopwords, punctuation, spaces
- Removes tokens shorter than 2 chars
- Lemmatizes: `"rendering"` → `"render"`, `"scrollbars"` → `"scrollbar"`
- Extracts named entities (used for metadata, not retrieval)

#### Unified Text Assembly
```python
unified_text = " . ".join([clean_title, clean_body, "Labels: ...", "Comments: ..."])
```
Comments: only first 2 comments included (noise reduction).

#### Output stored in `test_processed.json`
Each record contains: `clean_title`, `clean_body`, `unified_text`, `tokens[]`, `entities[]`, `token_count`.

---

## 3. Stage 1 — Embedding & Indexing

**Module:** `modules/embedding.py`, `modules/vector_store.py`

### Dense Embeddings

**Model:** `BAAI/bge-large-en`
- Architecture: BERT-based bi-encoder
- Output dimension: **1024**
- Normalization: L2-normalized (cosine similarity = dot product)
- Parameters: ~335M

**Query-time instruction prefix** (BGE-specific):
```
"Represent this sentence for searching relevant passages: " + query
```
This prefix is prepended to queries only (not documents) — it's required by BGE models to align the embedding space between queries and passages.

**Storage:** ChromaDB (local persistent vector store)
- Collection: `ticket_embeddings`
- Distance metric: `cosine`
- Each document stored as: `(id, embedding[1024], document_text, metadata{})`

### Sparse Index (BM25)

**Library:** `rank_bm25.BM25Okapi`
- Algorithm: BM25+ variant
- Built in-memory at first use, cached as module-level globals
- Rebuilt when `rebuild_bm25_index()` is called

**Title Boosting (Improvement 3):**
```python
boosted = f"{title} {title} {doc}"   # title appears 3× total
```
Why: Default `unified_text` is ~500 words; title is ~8 words. Without boosting, title TF contribution is only ~5%. Boosting raises it to ~15%, better reflecting that titles are highly discriminative for routing.

**Tokenization:** `text.lower().split()` on the boosted corpus text.

---

## 4. Stage 2 — Hybrid Retrieval

**Module:** `modules/retrieval.py`

### Dense Retrieval

ChromaDB `query()` with L2 distance metric:
```
similarity = 1 - L2_distance
```
Returns top-**30** results sorted by cosine similarity.

### Sparse Retrieval (BM25)

**Query tokenization (Improvement 4 — spaCy expansion):**
```python
clean_q = clean_text_regex(query)
spacy_result = process_with_spacy(clean_q)
tokenized_query = spacy_result["tokens"]   # lemmatized, stopwords removed
```
This aligns query tokenization with how corpus documents were preprocessed — preventing vocabulary mismatch where query uses "rendering" but BM25 only has "render".

BM25 scoring for each document `d`:
```
BM25(d, q) = Σ_term IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1 × (1 - b + b × |d|/avgdl))
```
Parameters: `k1=1.5, b=0.75` (BM25Okapi defaults). Returns top-**30** results.

### Weighted Reciprocal Rank Fusion (Improvement 2)

Standard RRF (Cormack et al., 2009) extended with per-source weights:

```
WRRF_score(d) = dense_weight  × Σ_dense_rank  1 / (k + rank + 1)
              + sparse_weight × Σ_sparse_rank 1 / (k + rank + 1)
```

| Parameter | Value | Why |
|-----------|-------|-----|
| `k` | 60 | Standard RRF smoothing constant |
| `dense_weight` | **1.2** | BAAI/bge-large-en (1024-dim) carries stronger semantic signal |
| `sparse_weight` | **1.0** | BM25 is strong for keyword-exact queries |

Documents appearing in both lists get contribution from both terms → natural fusion bonus for overlap items.

Output: **top-60 candidates** sorted by WRRF score, passed to the reranker.

---

## 5. Stage 3 — Cross-Encoder Reranking

**Module:** `modules/reranker.py`  
**Model:** `BAAI/bge-reranker-large`

### What a cross-encoder does differently from bi-encoder

| | Bi-encoder (embedding) | Cross-encoder (reranker) |
|---|---|---|
| Input | Encodes query and doc **separately** | Processes (query, doc) **together** |
| Attention | Query self-attention only | Full cross-attention between query and doc |
| Speed | Fast (precomputed embeddings) | Slow (must run for every pair) |
| Accuracy | Good approximation | Higher precision |
| Use case | First-stage retrieval (millions of docs) | Second-stage reranking (10-60 candidates) |

### Process

```python
pairs = [(query, doc_text) for doc in top_60_candidates]
scores = cross_encoder.predict(pairs)   # sigmoid(logit) per pair
```

**Model:** `BAAI/bge-reranker-large`
- Parameters: ~340M (vs 110M for `bge-reranker-base`)
- Trained on MS MARCO, BEIR benchmarks
- Output: relevance score in `[-10, 10]` range (raw logit)

Pairs are sorted descending by score. Top-**10** are returned with `rerank_score` field added.

These top-10 are what the classifier sees as `retrieved_docs`.

---

## 6. Stage 4 — Retrieval-Augmented Classification (RAC)

**Module:** `modules/classifier.py`  
**Model:** `Ollama llama3.1:8b` (local inference)

### Two Classification Modes

#### Generic Mode (no `known_teams`)
Used by `evaluation/evaluate.py` (the simple evaluation module). Classifies into:
`["frontend", "backend", "infrastructure"]`

#### Fine-Grained Mode (with `known_teams` + `retrieved_docs`)
Used by `tests/eval_engine.py` and `pipeline/triage_pipeline.py`.

`known_teams` is built at eval time:
```python
known_teams = unique_preserve_order(
    [normalize_label(issue["team"]) for issue in test_issues]
    + list(corpus_team_counter.keys())
)
```
This gives ~40-80 fine-grained team names like `"css parsing and computation"`, `"html parser"`, `"selection"`, etc.

### Prompt Engineering (Fine-Grained)

The full prompt has 4 sections:

**Section 1 — Strict rules** (JSON output enforcement):
```
STRICT RULES:
1. Respond with ONLY valid JSON — no markdown, no explanation.
2. JSON must have exactly: "type", "severity", "team"
```

**Section 2 — Valid teams list:**
```
VALID TEAMS (you MUST pick exactly one):
  - css parsing and computation
  - html parser
  - selection
  ... (all ~60+ teams)
```

**Section 3 — Team Frequency Summary (from retrieved docs):**
```
TEAM FREQUENCY SUMMARY (across top-10 similar tickets):
  html parser                         ████ (4/10)
  css parsing and computation         ███  (3/10)
  selection                           ██   (2/10)
  → Frequency hint: 'html parser' appears most often (4/10)
    — consider this as one signal alongside the ticket content.
```
*Note: ⭐ hint only shown when top team has ≥40% (≥4/10) frequency. When spread evenly: "Teams are spread — rely on ticket content."*

**Section 4 — Individual similar tickets (top-10):**
```
SIMILAR RESOLVED TICKETS:
  #1: "CSS overflow: scrollbar not rendered" → Team: css parsing | Component: layout (relevance: 8.423)
  #2: "Scrollbar visibility in overflow-x containers" → Team: css parsing | ...
  ...
```

**Chain-of-thought reasoning:**
```
Think step-by-step:
1. What is this ticket about? (core topic, component, keywords)
2. Which similar tickets match most closely by topic?
3. What team handled those closest matches?
4. Does the frequency summary agree?
5. Therefore, the correct team is...
```

### Post-Processing & Fallback

If LLM outputs a team not in `known_teams`:
1. **Exact match** (case-insensitive) — accept
2. **Substring match** — e.g. LLM says `"html"` → matches `"html parser"`
3. **Most frequent retrieved team** — use the team that appears most in top-10 docs
4. **Last resort** — `known_teams[0]`

If all 3 retries fail:
- Fallback = most prevalent team in retrieved docs (not hardcoded `"backend"`)

**LLM Settings:**
- Temperature: `0.1` (near-deterministic for consistent classification)
- Max tokens: `300`
- Retries: up to 3

---

## 7. Stage 4b — HyDE Fallback

**Module:** `modules/hyde.py`  
**Trigger:** Classifier-retrieval agreement < 0.3

### Agreement Calculation

```python
retrieved_team_list = [normalize(doc["metadata"]["team"]) for doc in reranked_docs[:10]]
pred_team = normalize(classification["team"])
agreement = count(pred_team in retrieved_team_list) / len(retrieved_team_list)

if agreement < 0.3:
    # trigger HyDE
```

Example: if the top-10 retrieved docs have `[layout, layout, layout, editor, tables, ...]` and the classifier predicted `"html parser"`, agreement = 0/10 = 0.0 → HyDE fires.

### Adaptive HyDE Generation (Improvement 6)

Query type is detected first:
```python
# Feature signal keywords:
"add", "support", "implement", "new", "feature", "allow", "enable", ...

# Improvement signal keywords:
"improve", "enhance", "performance", "optim", "better", "faster", "slow", ...

# Default: bug
```

Then the matching prompt template is used:
- **Bug**: "write a hypothetical resolved bug report with root cause + fix"
- **Feature**: "write a hypothetical feature implementation discussion"
- **Improvement**: "write a hypothetical performance improvement analysis"

The hypothetical document is embedded **without** the query instruction prefix (it's treated as a document, not a query):
```python
hyde_embedding = generate_embedding(hypothetical_doc, is_query=False)
```

Then:
1. Dense retrieval with hyde embedding (top-20)
2. BM25 retrieval with hypothetical text (top-20)
3. Weighted RRF fusion → top-20
4. Rerank vs original query
5. If `hyde_confidence > original_confidence` → use HyDE results
6. Re-classify with new retrieved docs

---

## 8. Stage 5 — RAG Response Generation

**Module:** `modules/generator.py`  
**Model:** `Ollama llama3.1:8b`

The top-**5** reranked documents (after potential HyDE replacement) are formatted as context:

```
SIMILAR RESOLVED ISSUES:
Issue 1:
Title: <title>
Team: <team>  Component: <component>
Description: <first 400 chars of document text>
---
Issue 2:
...
```

The generation prompt instructs the model to:
- Acknowledge the ticket type and severity
- Reference the most similar resolved issues
- Provide concrete debugging steps and fix directions
- Suggest which code areas/components to investigate

**LLM Settings:**
- Temperature: `0.3` (slightly creative for actionable suggestions)
- Max tokens: `800`

---

## 9. Evaluation Framework

**Module:** `tests/eval_engine.py`

### Classification Metrics

Computed using `sklearn.metrics` on 100 test queries.

#### Top-K Accuracy
```python
# For each query:
top1_hit = (predicted_team == ground_truth_team)
top3_hit = (ground_truth_team in top3_predicted_teams)
top5_hit = (ground_truth_team in top5_predicted_teams)

top1_accuracy = sum(top1_hits) / 100
```
**What it measures:** How often the classifier's #1 prediction exactly matches the actual team the ticket was assigned to.

#### Precision, Recall, F1 (Weighted)
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall    = recall_score   (y_true, y_pred, average="weighted", zero_division=0)
f1        = f1_score       (y_true, y_pred, average="weighted", zero_division=0)
```
- `average="weighted"`: each team's score is weighted by its frequency in `y_true`
- `zero_division=0`: teams with no predictions contribute 0 precision (not error)

**What they measure:**
- **Precision (cls):** Of all tickets predicted as team X, what fraction actually belonged to X?
- **Recall (cls):** Of all tickets that belong to team X, what fraction did we predict as X?
- **F1 (cls):** Harmonic mean of precision and recall

---

### Retrieval Metrics

Computed over the top-K retrieved documents per query.

#### Ground Truth Construction
A retrieved document is considered **relevant** if it shares at least one label with the query ticket:
```python
relevant = [
    doc for doc in retrieved_docs
    if set(doc["metadata"]["labels"].split(", ")) & set(issue["labels"])
]
```

#### Precision@K
```
P@K = (# relevant documents in top-K) / K
```
**Library:** Custom `precision_at_k()` in `eval_engine.py`  
**Example:** If 2 of 5 retrieved docs are relevant → P@5 = 0.4

#### Recall@K
```
R@K = (# relevant in top-K) / (# total relevant in corpus)
```
**Why recall@k is low (0.03):** Each team can have hundreds of relevant documents. Retrieving 5 out of 200+ relevant docs gives ~2.5% recall. This metric is structurally low for large corpora — it's not a sign of poor retrieval.

#### Mean Reciprocal Rank (MRR)
```
RR = 1 / rank_of_first_relevant_document
MRR = mean(RR) over all queries
```
**Library:** Custom `reciprocal_rank()` in `eval_engine.py`  
**Example:** First relevant doc at rank 2 → RR = 0.5  
**What it measures:** How high up does the first correct answer appear?

#### NDCG@K (Normalized Discounted Cumulative Gain)
```
DCG@K  = Σ_{i=1}^{K}  rel_i / log2(i+1)
IDCG@K = Σ_{i=1}^{min(K,R)}  1 / log2(i+1)   (ideal = all relevant at top)
NDCG@K = DCG@K / IDCG@K
```
**Library:** Custom `ndcg_at_k()` in `eval_engine.py` using binary relevance  
**What it measures:** Quality of result ranking — relevant items near the top score higher than relevant items near the bottom.

---

### Evaluation Modes

| Mode | Reranker | HyDE | Classification | `use_reranker` | `hyde_enabled` |
|------|----------|------|---------------|----------------|---------------|
| Dense baseline | ❌ | ❌ | Fine-grained | `False` | `False` |
| Sparse baseline | ❌ | ❌ | Fine-grained | `False` | `False` |
| **Hybrid (full)** | ✅ | ✅ | Fine-grained | `True` | `True` |

---

## 10. Configuration Reference

All values are in `.env` and read by `config/settings.py`.

| Setting | Default | Description |
|---------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.1:8b` | Local LLM for classification + generation |
| `OLLAMA_TIMEOUT_SECONDS` | `120` | Timeout per LLM call |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en` | Bi-encoder embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-large` | Cross-encoder reranker model |
| `RETRIEVAL_TOP_K` | `30` | Dense retrieval candidate count |
| `BM25_TOP_K` | `30` | Sparse retrieval candidate count |
| `RERANK_TOP_N` | `5` | Final top-N after reranking (eval uses top-10 for RAC) |
| `RRF_K` | `60` | RRF smoothing constant |
| `DENSE_WEIGHT` | `1.2` | Dense results weight in weighted RRF |
| `SPARSE_WEIGHT` | `1.0` | Sparse results weight in weighted RRF |
| `HYDE_ENABLED` | `true` | Whether HyDE is active (hybrid only) |
| `HYDE_CONFIDENCE_THRESHOLD` | `0.3` | Reranker score fallback threshold |
| `HYDE_AGREEMENT_THRESHOLD` | `0.3` | Classifier-retrieval agreement threshold |
| `MAX_ISSUES` | `3000` | Max issues fetched from data source |
| `TEST_SPLIT_COUNT` | `100` | Number of issues in test split |

---

## Model Summary

| Role | Model | Params | Framework |
|------|-------|--------|-----------|
| Embedding | `BAAI/bge-large-en` | 335M | sentence-transformers |
| Sparse indexing | BM25Okapi | — | rank-bm25 |
| Reranking | `BAAI/bge-reranker-large` | 340M | sentence-transformers CrossEncoder |
| Classification | `llama3.1:8b` | 8B | Ollama (local) |
| HyDE generation | `llama3.1:8b` | 8B | Ollama (local) |
| RAG generation | `llama3.1:8b` | 8B | Ollama (local) |
| NLP preprocessing | `en_core_web_sm` | 12M | spaCy |
