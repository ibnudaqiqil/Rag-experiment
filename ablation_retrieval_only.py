# ablation_retrieval_only.py
from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from langchain.schema import Document

from bm25_index import BM25Index
from reranker import rerank_crossencoder
from utils import parse_gold_set, normalize_tag, normalize_tag_list, rank_metrics


@dataclass
class AblationRetrieveOnly:
    strategies: Sequence[str] = (
        "BM25 (standar)",
        "BM25 + MMR",
        "BM25 Page-Group",
        "BM25 + Multi-Query",
        "BM25 + HyDE",
        "BM25 + Keyword Must",
    )
    reranker_flags: Sequence[str] = ("ON", "OFF")
    k_list: Sequence[int] = (5, 10)
    # params strategi
    mmr_lambda_list: Sequence[float] = (0.5,)
    mmr_fetch_k_list: Sequence[int] = (48,)
    parent_merge_chars_list: Sequence[int] = (2400,)
    num_expansions_list: Sequence[int] = (4,)
    must_keywords_list: Sequence[str] = ("",)

    # kolom & normalisasi label
    col_question: str = "question"
    col_gold_label: str = "gold_label"
    normalize_labels: bool = True
    include_page_suffix: Optional[bool] = False


def _rrf_merge(rank_lists, k_rrf: int = 60):
    pool = {}
    for lst in rank_lists:
        for rank, (d, sc) in enumerate(lst, 1):
            key = id(d)
            if key not in pool:
                pool[key] = {"doc": d, "bm25_best": float(sc), "rrf": 0.0}
            pool[key]["bm25_best"] = max(pool[key]["bm25_best"], float(sc))
            pool[key]["rrf"] += 1.0/(k_rrf+rank)
    merged = [(v["doc"], v["bm25_best"], v["rrf"]) for v in pool.values()]
    merged.sort(key=lambda x: x[2], reverse=True)
    return merged


def _apply_must_keywords(docs_sc, must_terms):
    if not must_terms:
        return docs_sc
    must = [t.strip().lower() for t in must_terms if t.strip()]
    if not must:
        return docs_sc
    out = []
    for d, sc in docs_sc:
        if all(t in (d.page_content or "").lower() for t in must):
            out.append((d, sc))
    return out or docs_sc


def _page_group_merge(chosen, max_chars: int):
    from collections import defaultdict
    from langchain.schema import Document
    groups = defaultdict(list)
    for d, _ in chosen:
        m = d.metadata or {}
        src = str(m.get("source", "doc"))
        pg = int(m.get("page", -1)) if m.get("page") is not None else -1
        groups[(src, pg)].append(d)
    out = []
    for (src, pg), lst in groups.items():
        lst.sort(key=lambda x: (x.metadata or {}).get("start_index", 0))
        acc = ""
        for d in lst:
            t = d.page_content or ""
            if len(acc)+len(t)+1 > max_chars:
                break
            acc += (("\n" if acc else "") + t)
        out.append(Document(page_content=acc, metadata={
                   "source": src, "page": None if pg == -1 else pg}))
    return out


def _expand_multi_query(llm, q: str, n: int):
    # retrieve-only -> tidak pakai LLM; kembalikan varian minimal
    return [q]  # bisa diganti generator lokal jika perlu


def _expand_hyde(llm, q: str):
    # retrieve-only -> tidak pakai LLM
    return q


def _retrieve_by_strategy(strategy: str, q: str, index: BM25Index,
                          k_final: int, use_reranker: bool,
                          mmr_lambda: float, mmr_fetch_k: int,
                          parent_merge_chars: int, num_expansions: int,
                          must_keywords: str):
    pre_k = max(k_final, mmr_fetch_k)

    if strategy == "BM25 (standar)":
        base = index.search(q, topn=pre_k)
        docs = [d for d, _ in base]

    elif strategy == "BM25 + MMR":
        mmr_list = index.mmr(q, k=pre_k, fetch_k=max(
            pre_k, mmr_fetch_k), lam=mmr_lambda)
        docs = [d for d, _ in mmr_list]

    elif strategy == "BM25 Page-Group":
        base = index.search(q, topn=pre_k)
        docs = _page_group_merge(base, parent_merge_chars)[:pre_k]

    elif strategy == "BM25 + Multi-Query":
        variants = _expand_multi_query(None, q, num_expansions)
        lists = [index.search(v, topn=pre_k) for v in variants]
        merged = _rrf_merge(lists)
        docs = [d for d, _, _ in merged[:pre_k]]

    elif strategy == "BM25 + HyDE":
        pseudo = _expand_hyde(None, q)
        base = index.search(pseudo, topn=pre_k)
        docs = [d for d, _ in base]

    elif strategy == "BM25 + Keyword Must":
        base = index.search(q, topn=max(pre_k, mmr_fetch_k))
        docs = [d for d, _ in _apply_must_keywords(
            base, (must_keywords or "").split(","))][:pre_k]

    else:
        docs = [d for d, _ in index.search(q, topn=pre_k)]

    if use_reranker and docs:
        docs, _ = rerank_crossencoder(q, docs, top_k=k_final)
    else:
        docs = docs[:k_final]
    return docs


def run_ablation_retrieval_only(df_eval: pd.DataFrame, index: BM25Index, setup: AblationRetrieveOnly):
    if setup.col_question not in df_eval.columns:
        raise ValueError(f"CSV must contain column '{setup.col_question}'.")

    detail_rows, summary_rows = [], []

    grid = product(
        setup.strategies,
        setup.k_list,
        setup.reranker_flags,
        setup.mmr_lambda_list,
        setup.mmr_fetch_k_list,
        setup.parent_merge_chars_list,
        setup.num_expansions_list,
        setup.must_keywords_list,
    )

    for (strategy, k_final, rr_flag, lam, fk, pmc, n_exp, must_kw) in grid:
        rr_on = (str(rr_flag).upper() == "ON")
        rows, t_list = [], []

        for _, r in df_eval.iterrows():
            q = str(r.get(setup.col_question, "")).strip()
            gold = str(r.get(setup.col_gold_label, "")).strip() if (
                setup.col_gold_label in df_eval.columns and not pd.isna(r.get(setup.col_gold_label, ""))) else ""

            import time
            t0 = time.perf_counter()
            docs = _retrieve_by_strategy(strategy, q, index,
                                         k_final=int(k_final), use_reranker=rr_on,
                                         mmr_lambda=float(lam), mmr_fetch_k=int(fk),
                                         parent_merge_chars=int(pmc), num_expansions=int(n_exp),
                                         must_keywords=str(must_kw or ""))
            t1 = time.perf_counter()

            # top@k tag
            top_raw = []
            for d in docs[:int(k_final)]:
                m = d.metadata or {}
                src = str(m.get("source", ""))
                page = m.get("page", None)
                tag = f"{src}:p{page}" if (page is not None) else src
                top_raw.append(tag)

            # normalisasi label
            if setup.normalize_labels:
                top_norm = normalize_tag_list(
                    top_raw, include_page_suffix=setup.include_page_suffix)
                gold_set = {normalize_tag(
                    g, include_page_suffix=setup.include_page_suffix) for g in parse_gold_set(gold)}
            else:
                top_norm = top_raw[:]
                gold_set = parse_gold_set(gold)

            metr = rank_metrics(top_norm, gold_set, k=int(k_final)) if gold_set else {
                f"hits@{k_final}": 0.0,
                f"precision@{k_final}": float("nan"),
                f"recall@{k_final}": float("nan"),
                f"f1@{k_final}": float("nan"),
                "rr": float("nan"), "ap": float("nan"),
                f"ndcg@{k_final}": float("nan"),
            }

            t_list.append(t1-t0)
            rows.append({
                "question": q,
                "strategy": strategy,
                "k": int(k_final),
                "reranker": "ON" if rr_on else "OFF",
                "mmr_lambda": float(lam) if "MMR" in strategy else None,
                "mmr_fetch_k": int(fk) if ("MMR" in strategy or "Keyword" in strategy) else None,
                "parent_merge_chars": int(pmc) if "Page-Group" in strategy else None,
                "num_expansions": int(n_exp) if "Multi-Query" in strategy else None,
                "must_keywords": str(must_kw) if "Keyword" in strategy else None,
                f"retrieved_sources@{k_final}": "; ".join(top_raw),
                "retrieval_time_s": t1-t0,
                **metr
            })

        detail = pd.DataFrame(rows)
        detail_rows.append(detail)
        summary_rows.append({
            "strategy": strategy,
            "reranker": "ON" if rr_on else "OFF",
            "k": int(k_final),
            "mmr_lambda": float(lam) if "MMR" in strategy else None,
            "mmr_fetch_k": int(fk) if ("MMR" in strategy or "Keyword" in strategy) else None,
            "parent_merge_chars": int(pmc) if "Page-Group" in strategy else None,
            "num_expansions": int(n_exp) if "Multi-Query" in strategy else None,
            "must_keywords": str(must_kw) if "Keyword" in strategy else None,
            f"hits@{k_final}_avg": float(np.nanmean(detail.get(f"hits@{k_final}", np.nan))) if len(detail) else np.nan,
            f"precision@{k_final}_avg": float(np.nanmean(detail.get(f"precision@{k_final}", np.nan))) if len(detail) else np.nan,
            f"recall@{k_final}_avg": float(np.nanmean(detail.get(f"recall@{k_final}", np.nan))) if len(detail) else np.nan,
            f"f1@{k_final}_avg": float(np.nanmean(detail.get(f"f1@{k_final}", np.nan))) if len(detail) else np.nan,
            "mrr": float(np.nanmean(detail.get("rr", np.nan))) if len(detail) else np.nan,
            "map": float(np.nanmean(detail.get("ap", np.nan))) if len(detail) else np.nan,
            f"ndcg@{k_final}_avg": float(np.nanmean(detail.get(f"ndcg@{k_final}", np.nan))) if len(detail) else np.nan,
            "retrieval_time_avg_s": float(np.nanmean(detail.get("retrieval_time_s", np.nan))) if len(detail) else np.nan,
            "n_samples": int(len(detail)),
        })

    return (pd.concat(detail_rows, ignore_index=True) if detail_rows else pd.DataFrame(),
            pd.DataFrame(summary_rows))
