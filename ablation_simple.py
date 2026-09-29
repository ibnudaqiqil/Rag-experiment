# ablation_simple.py
# ---------------------------------------------------
# Simple ablation (RETRIEVE-ONLY) + progress callback.
# Setup yang diuji:
#  1) BM25 (Top-N)
#  2) BM25 + Multi-Query (RRF)
#  3) BM25 + HyDE
#  4) BM25 + Multi-Query + Rerank
#  5) BM25 + HyDE + Rerank
# ---------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from langchain.schema import Document, HumanMessage

from bm25_index import BM25Index
from reranker import rerank_crossencoder
from utils import parse_gold_set, normalize_tag, normalize_tag_list, rank_metrics

ProgressCB = Optional[Callable[[int, int, str, int, str], None]]
# callback signature: (done, total, setup_name, row_idx, question)


@dataclass
class SimpleAblationConfig:
    k: int = 10
    num_expansions: int = 4          # untuk Multi-Query
    include_page_suffix: bool = False
    normalize_labels: bool = True
    col_question: str = "question"
    col_gold_label: str = "gold_label"

# ----------------- Helpers -----------------


def _rrf_merge(rank_lists: List[List[Tuple[Document, float]]], k_rrf: int = 60):
    pool: Dict[int, Dict[str, float | Document]] = {}
    for lst in rank_lists:
        for rank, (d, sc) in enumerate(lst, 1):
            key = id(d)
            if key not in pool:
                pool[key] = {"doc": d, "bm25_best": float(sc), "rrf": 0.0}
            pool[key]["bm25_best"] = max(
                pool[key]["bm25_best"], float(sc))  # type: ignore[index]
            pool[key]["rrf"] = float(pool[key]["rrf"]) + \
                1.0/(k_rrf+rank)    # type: ignore[index]
    merged = [(v["doc"], float(v["bm25_best"]), float(v["rrf"]))
              for v in pool.values()]  # type: ignore[list-item]
    merged.sort(key=lambda x: x[2], reverse=True)
    return merged


def _expand_multi_query(llm: Optional[Any], query: str, n: int) -> List[str]:
    if llm is None or n <= 1:
        return [query]
    prompt = (
        "Generate {n} diverse reformulations of the search query for document retrieval.\n"
        "Return one per line without numbering.\n\nQuery: {q}"
    )
    res = llm([HumanMessage(content=prompt.format(n=n, q=query))])
    content = getattr(res, "content", str(res))
    lines = [ln.strip(" -•\t") for ln in content.splitlines() if ln.strip()]
    uniq = list(dict.fromkeys(lines))
    return [query] + uniq[: max(0, n-1)]


def _expand_hyde(llm: Optional[Any], query: str) -> str:
    if llm is None:
        return query
    prompt = (
        "Write a short (<=120 words), neutral pseudo-answer to the following query.\n"
        "No citations. Keep it factual and generic.\n\nQuery: {q}"
    )
    res = llm([HumanMessage(content=prompt.format(q=query))])
    return getattr(res, "content", str(res)).strip() or query


def _topk_tags(docs: List[Document], k: int) -> List[str]:
    out = []
    for d in docs[:k]:
        m = d.metadata or {}
        src = str(m.get("source", ""))
        page = m.get("page", None)
        tag = f"{src}:p{page}" if (page is not None) else src
        out.append(tag)
    return out


def _score_row(top_raw: List[str], gold: str, *, k: int, normalize_labels: bool, include_page_suffix: bool):
    if normalize_labels:
        top_norm = normalize_tag_list(
            top_raw, include_page_suffix=include_page_suffix)
        gold_set = {normalize_tag(
            g, include_page_suffix=include_page_suffix) for g in parse_gold_set(gold)}
    else:
        top_norm = top_raw[:]
        gold_set = parse_gold_set(gold)

    metr = rank_metrics(top_norm, gold_set, k=k) if gold_set else {
        f"hits@{k}": 0.0, f"precision@{k}": float("nan"), f"recall@{k}": float("nan"),
        f"f1@{k}": float("nan"), "rr": float("nan"), "ap": float("nan"), f"ndcg@{k}": float("nan")
    }
    return metr

# ----------------- Core Strategies -----------------


def _bm25_topn(q: str, index: BM25Index, k: int, *, rerank: bool) -> List[Document]:
    base = index.search(q, topn=max(k, 32))
    docs = [d for d, _ in base]
    if rerank and docs:
        docs, _ = rerank_crossencoder(q, docs, top_k=k)
    else:
        docs = docs[:k]
    return docs


def _bm25_multiq_rrf(q: str, index: BM25Index, k: int, *, llm: Optional[Any], n_exp: int, rerank: bool) -> List[Document]:
    variants = _expand_multi_query(llm, q, n_exp)
    lists = [index.search(v, topn=max(k, 32)) for v in variants]
    merged = _rrf_merge(lists)
    docs = [d for d, _, _ in merged[:max(k, 32)]]
    if rerank and docs:
        docs, _ = rerank_crossencoder(q, docs, top_k=k)
    else:
        docs = docs[:k]
    return docs


def _bm25_hyde(q: str, index: BM25Index, k: int, *, llm: Optional[Any], rerank: bool) -> List[Document]:
    pseudo = _expand_hyde(llm, q)
    base = index.search(pseudo, topn=max(k, 32))
    docs = [d for d, _ in base]
    if rerank and docs:
        docs, _ = rerank_crossencoder(q, docs, top_k=k)
    else:
        docs = docs[:k]
    return docs

# ----------------- Public Runner (+progress) -----------------


def run_simple_ablation(
    df_eval: pd.DataFrame,
    index: BM25Index,
    *,
    config: SimpleAblationConfig,
    llm: Optional[Any] = None,
    progress_cb: ProgressCB = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Jalankan 5 konfigurasi sederhana (retrieve-only).
    progress_cb(d, total, setup_name, row_idx, question) dipanggil setiap sample.
    """
    k = int(config.k)
    if config.col_question not in df_eval.columns:
        raise ValueError(f"CSV must contain column '{config.col_question}'.")

    setups: List[Tuple[str, Any]] = [
        ("BM25 (Top-N)", lambda q: _bm25_topn(q, index, k, rerank=False)),
        ("BM25 + Multi-Query (RRF)", lambda q: _bm25_multiq_rrf(q,
         index, k, llm=llm, n_exp=config.num_expansions, rerank=False)),
        ("BM25 + HyDE", lambda q: _bm25_hyde(q, index, k, llm=llm, rerank=False)),
        ("BM25 + Multi-Query + Rerank", lambda q: _bm25_multiq_rrf(q,
         index, k, llm=llm, n_exp=config.num_expansions, rerank=True)),
        ("BM25 + HyDE + Rerank", lambda q: _bm25_hyde(q, index, k, llm=llm, rerank=True)),
    ]

    total = len(df_eval) * len(setups)
    done = 0

    details: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []

    for name, retr_fn in setups:
        rows, t_list = [], []
        for ridx, r in df_eval.iterrows():
            q = str(r.get(config.col_question, "")).strip()
            gold = str(r.get(config.col_gold_label, "")).strip() if (
                config.col_gold_label in df_eval.columns and not pd.isna(r.get(config.col_gold_label, ""))) else ""

            # progress callback
            done += 1
            if progress_cb:
                try:
                    progress_cb(done, total, name, ridx, q)
                except Exception:
                    pass

            import time
            t0 = time.perf_counter()
            docs = retr_fn(q)
            t1 = time.perf_counter()

            top_raw = _topk_tags(docs, k)
            metr = _score_row(top_raw, gold, k=k,
                              normalize_labels=config.normalize_labels,
                              include_page_suffix=config.include_page_suffix)

            rows.append({
                "question": q,
                "setup": name,
                f"retrieved_sources@{k}": "; ".join(top_raw),
                "retrieval_time_s": t1 - t0,
                **metr,
            })
            t_list.append(t1 - t0)

        detail = pd.DataFrame(rows)
        details.append(detail)
        summaries.append({
            "setup": name,
            f"hits@{k}_avg": float(np.nanmean(detail.get(f"hits@{k}", np.nan))) if len(detail) else np.nan,
            f"precision@{k}_avg": float(np.nanmean(detail.get(f"precision@{k}", np.nan))) if len(detail) else np.nan,
            f"recall@{k}_avg": float(np.nanmean(detail.get(f"recall@{k}", np.nan))) if len(detail) else np.nan,
            f"f1@{k}_avg": float(np.nanmean(detail.get(f"f1@{k}", np.nan))) if len(detail) else np.nan,
            "mrr": float(np.nanmean(detail.get("rr", np.nan))) if len(detail) else np.nan,
            "map": float(np.nanmean(detail.get("ap", np.nan))) if len(detail) else np.nan,
            f"ndcg@{k}_avg": float(np.nanmean(detail.get(f"ndcg@{k}", np.nan))) if len(detail) else np.nan,
            "retrieval_time_avg_s": float(np.nanmean(detail.get("retrieval_time_s", np.nan))) if len(detail) else np.nan,
            "n_samples": int(len(detail)),
        })

    detail_all = pd.concat(
        details, ignore_index=True) if details else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    return detail_all, summary_df


__all__ = ["SimpleAblationConfig", "run_simple_ablation"]
