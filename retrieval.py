# retrieval.py
# ---------------------------------------------------
# Retrieval handler untuk BM25 dan reranker.
# ---------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Optional, Any
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd

from bm25_index import BM25Index
from utils import rank_metrics, compute_answer_match
from reranker import rerank_crossencoder
from langchain.schema import Document

# ---------- Retrieval ----------


def retrieve(
    query: str,
    index: BM25Index,
    llm: Optional[Any] = None,
    k_docs: int = 10,
    use_reranker: bool = False,
    return_debug: bool = False
) -> Tuple[List[Document], Optional[pd.DataFrame]]:
    """
    Retrieve relevant documents based on BM25 and (optionally) reranker.
    """
    pre_k = k_docs
    # --- 1. Retrieve basic BM25 results ---
    bm25_results = index.search(query, topn=pre_k)

    docs = [d for d, _ in bm25_results]

    # --- 2. Apply reranker (if enabled) ---
    if use_reranker:
        docs, rerank_scores = rerank_crossencoder(query, docs, top_k=k_docs)

    # --- 3. Optional debug data ---
    debug_df = None
    if return_debug:
        debug_data = []
        for i, doc in enumerate(docs):
            debug_data.append({
                "rank": i + 1,
                "doc_id": doc.metadata.get("source", "unknown"),
                "content": doc.page_content[:200],  # preview of the content
            })
        debug_df = pd.DataFrame(debug_data)

    return docs, debug_df

# ---------- Evaluation of answers ----------


def evaluate_answer(
    pred: str,
    ref: str,
    cos_sim: Optional[float] = None,
    use_em: bool = True,
    cosine_thr: float = 0.82,
    f1_thr: float = 0.80
) -> Dict[str, float]:
    """
    Evaluates an answer's match to the reference answer using exact match, f1, and cosine similarity.
    """
    return compute_answer_match(pred, ref, cos_sim, use_em, cosine_thr, f1_thr)

# ---------- Evaluation metrics ----------


def evaluate_retrieval(
    query: str,
    gold_labels: List[str],
    index: BM25Index,
    k_docs: int = 10,
    use_reranker: bool = False
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality based on rank metrics (e.g., Hits@k, Precision@k, Recall@k, F1-score).
    """
    docs, _ = retrieve(query, index, k_docs=k_docs, use_reranker=use_reranker)
    top_sources = [doc.metadata.get("source", "unknown") for doc in docs]
    gold_set = set(gold_labels)

    # Calculate rank metrics (e.g., hits@k, precision@k, etc.)
    rank_metrics_result = rank_metrics(top_sources, gold_set, k=k_docs)

    return rank_metrics_result


__all__ = ["retrieve", "evaluate_answer", "evaluate_retrieval"]
