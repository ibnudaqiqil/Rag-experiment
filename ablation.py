# ablation.py (ADVANCED)
# ---------------------------------------------------
# Ablation komprehensif untuk membandingkan strategi:
#  - BM25 (Top-N)
#  - BM25 + MMR (diversity)
#  - BM25 Page-Group (parent-like)
#  - BM25 + Multi-Query (RRF merge)
#  - BM25 + HyDE (pseudo-answer)
#  - BM25 + Keyword Must (filter kandidat mengandung kata wajib)
# + opsi Reranker Cross-Encoder ON/OFF
# ---------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from langchain.schema import HumanMessage, SystemMessage, Document

from bm25_index import BM25Index
from reranker import rerank_crossencoder
import llm_cache
from utils import (
    parse_gold_set, normalize_tag, normalize_tag_list, rank_metrics,
)

# ====== Konfigurasi ======


@dataclass
class AblationSetup:
    # Ruang konfigurasi
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

    # Parameter strategi
    mmr_lambda_list: Sequence[float] = (0.5,)
    mmr_fetch_k_list: Sequence[int] = (48,)
    parent_merge_chars_list: Sequence[int] = (2400,)
    num_expansions_list: Sequence[int] = (4,)
    # koma-separate kata wajib, contoh: "aturan,kebijakan"
    must_keywords_list: Sequence[str] = ("",)

    # Normalisasi label untuk metrik
    normalize_labels: bool = True
    # default abaikan suffix ":pN" saat evaluasi
    include_page_suffix: Optional[bool] = False

    # Kolom CSV
    col_question: str = "question"
    col_gold_label: str = "gold_label"
    col_gold_mode: str = "gold_mode"

    # Output
    save_dir: Optional[str] = None
    debug_dump_path: Optional[str] = None  # simpan pasangan top@k vs gold

# ====== Util pendukung retrieval ======


def _rrf_merge(rank_lists: List[List[Tuple[Document, float]]], k_rrf: int = 60) -> List[Tuple[Document, float, float]]:
    """
    Reciprocal Rank Fusion untuk menggabungkan beberapa daftar (doc, bm25_score).
    Kembalikan list (doc, bm25_best, rrf_score) urut menurun rrf.
    """
    pool: Dict[int, Dict[str, float | Document]] = {}
    for lst in rank_lists:
        for rank, (d, sc) in enumerate(lst, 1):
            key = id(d)
            if key not in pool:
                pool[key] = {"doc": d, "bm25_best": float(sc), "rrf": 0.0}
            pool[key]["bm25_best"] = max(
                pool[key]["bm25_best"], float(sc))  # type: ignore[index]
            pool[key]["rrf"] = float(pool[key]["rrf"]) + \
                1.0 / (k_rrf + rank)  # type: ignore[index]
    merged = [(v["doc"], float(v["bm25_best"]), float(v["rrf"]))
              for v in pool.values()]  # type: ignore[list-item]
    merged.sort(key=lambda x: x[2], reverse=True)
    return merged


def _apply_must_keywords(docs_sc: List[Tuple[Document, float]], must_terms: List[str]) -> List[Tuple[Document, float]]:
    if not must_terms:
        return docs_sc
    must = [t.strip().lower() for t in must_terms if t.strip()]
    if not must:
        return docs_sc
    out = []
    for d, sc in docs_sc:
        text = (d.page_content or "").lower()
        if all(t in text for t in must):
            out.append((d, sc))
    return out or docs_sc  # fallback bila kosong


def _page_group_merge(chosen: List[Tuple[Document, float]], max_chars: int) -> List[Document]:
    """Gabungkan chunk (source,page) menjadi satu parent-like doc (urutan by start_index)."""
    from collections import defaultdict
    groups: Dict[Tuple[str, int], List[Document]] = defaultdict(list)
    for d, _ in chosen:
        m = d.metadata or {}
        src = str(m.get("source", "doc"))
        pg = int(m.get("page", -1)) if m.get("page") is not None else -1
        groups[(src, pg)].append(d)
    out_docs: List[Document] = []
    for (src, pg), lst in groups.items():
        lst.sort(key=lambda x: (x.metadata or {}).get("start_index", 0))
        acc = ""
        for d in lst:
            t = d.page_content or ""
            if len(acc) + len(t) + 1 > max_chars:
                break
            acc += (("\n" if acc else "") + t)
        out_docs.append(Document(page_content=acc, metadata={
                        "source": src, "page": None if pg == -1 else pg}))
    return out_docs


def _llm_id(llm) -> tuple[str, str]:
    """Identitas LLM untuk kunci cache: (provider, model)."""
    return (type(llm).__name__, str(getattr(llm, "model_name", None)
                                    or getattr(llm, "model", "?")))


def _expand_multi_query(llm, query: str, n: int) -> List[str]:
    if llm is None or n <= 1:
        return [query]
    prompt = (
        "Generate {n} diverse reformulations of the search query for document retrieval.\n"
        "Return one per line without numbering.\n\nQuery: {q}"
    )
    teks = prompt.format(n=n, q=query)
    provider, model = _llm_id(llm)
    content = llm_cache.ambil_atau_panggil(
        "multi_query", provider, model, query, n, teks,
        lambda: getattr(llm([HumanMessage(content=teks)]), "content", ""))
    if content is None:          # pemanggilan gagal -> jangan rusak eksperimen
        return [query]
    lines = [ln.strip(" -•\t") for ln in content.splitlines() if ln.strip()]
    if len(lines) < 1:
        lines = [query]
    # unik & jaga urutan
    uniq = list(dict.fromkeys(lines))
    return [query] + uniq[: max(0, n-1)]


def _expand_hyde(llm, query: str) -> str:
    if llm is None:
        return query
    prompt = (
        "Write a short (<=120 words), neutral pseudo-answer to the following query.\n"
        "No citations. Keep it factual and generic.\n\nQuery: {q}"
    )
    teks = prompt.format(q=query)
    provider, model = _llm_id(llm)
    content = llm_cache.ambil_atau_panggil(
        "hyde", provider, model, query, 1, teks,
        lambda: getattr(llm([HumanMessage(content=teks)]), "content", ""))
    return (content or "").strip() or query

# ====== Core retrieval untuk setiap strategi ======


def _retrieve_strategy(
    strategy: str,
    query: str,
    index: BM25Index,
    *,
    k_final: int,
    llm: Optional[Any],
    use_reranker: bool,
    mmr_lambda: float,
    mmr_fetch_k: int,
    parent_merge_chars: int,
    num_expansions: int,
    must_keywords: str,
) -> List[Document]:
    pre_k = max(k_final, mmr_fetch_k)

    if strategy == "BM25 (standar)":
        base = index.search(query, topn=pre_k)
        docs = [d for d, _ in base]

    elif strategy == "BM25 + MMR":
        mmr_list = index.mmr(query, k=pre_k, fetch_k=max(
            pre_k, mmr_fetch_k), lam=mmr_lambda)
        docs = [d for d, _ in mmr_list]

    elif strategy == "BM25 Page-Group":
        base = index.search(query, topn=pre_k)
        grouped = _page_group_merge(base, max_chars=parent_merge_chars)
        docs = grouped[:pre_k]

    elif strategy == "BM25 + Multi-Query":
        variants = _expand_multi_query(llm, query, n=num_expansions)
        lists = [index.search(qv, topn=pre_k) for qv in variants]
        merged = _rrf_merge(lists)
        docs = [d for d, _, _ in merged[:pre_k]]

    elif strategy == "BM25 + HyDE":
        pseudo = _expand_hyde(llm, query)
        base = index.search(pseudo, topn=pre_k)
        docs = [d for d, _ in base]

    elif strategy == "BM25 + Keyword Must":
        base = index.search(query, topn=max(pre_k, mmr_fetch_k))
        must_terms = [t.strip()
                      for t in (must_keywords or "").split(",") if t.strip()]
        filtered = _apply_must_keywords(base, must_terms)
        docs = [d for d, _ in filtered][:pre_k]

    else:
        # fallback
        base = index.search(query, topn=pre_k)
        docs = [d for d, _ in base]

    # Reranker opsional
    if use_reranker and docs:
        docs, _ = rerank_crossencoder(query, docs, top_k=k_final)
    else:
        docs = docs[:k_final]

    return docs

# ====== Jalankan 1 konfigurasi pada seluruh dataset ======


def _run_single_config(
    df_eval: pd.DataFrame,
    index: BM25Index,
    *,
    strategy: str,
    k_final: int,
    use_reranker: bool,
    llm: Optional[Any],
    # params strategi:
    mmr_lambda: float,
    mmr_fetch_k: int,
    parent_merge_chars: int,
    num_expansions: int,
    must_keywords: str,
    # kolom & normalisasi:
    col_question: str,
    col_gold_label: str,
    col_gold_mode: str = "gold_mode",
    normalize_labels: bool,
    include_page_suffix: Optional[bool],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows, t_retr, t_all = [], [], []

    for _, r in df_eval.iterrows():
        q = str(r.get(col_question, "")).strip()
        gold = str(r.get(col_gold_label, "")).strip() if (
            col_gold_label in df_eval.columns and not pd.isna(r.get(col_gold_label, ""))) else ""
        gold_mode = str(r.get(col_gold_mode, "all") or "all").strip() if (
            col_gold_mode in df_eval.columns) else "all"

        import time
        t0 = time.perf_counter()
        docs = _retrieve_strategy(
            strategy, q, index,
            k_final=k_final, llm=llm, use_reranker=use_reranker,
            mmr_lambda=mmr_lambda, mmr_fetch_k=mmr_fetch_k,
            parent_merge_chars=parent_merge_chars,
            num_expansions=num_expansions,
            must_keywords=must_keywords,
        )
        t1 = time.perf_counter()

        # top@k sources (raw tag: "src[:pN]")
        top_raw = []
        for d in docs[:k_final]:
            m = d.metadata or {}
            src = str(m.get("source", ""))
            page = m.get("page", None)
            tag = f"{src}:p{page}" if (page is not None) else src
            top_raw.append(tag)

        # normalisasi & metrik
        if normalize_labels:
            top_norm = normalize_tag_list(
                top_raw, include_page_suffix=include_page_suffix)
            gold_set = {normalize_tag(
                g, include_page_suffix=include_page_suffix) for g in parse_gold_set(gold)}
        else:
            top_norm = top_raw[:]
            gold_set = parse_gold_set(gold)

        metr = rank_metrics(top_norm, gold_set, k=k_final, gold_mode=gold_mode) if gold_set else {
            f"hits@{k_final}": 0.0,
            f"precision@{k_final}": float("nan"),
            f"recall@{k_final}": float("nan"),
            f"f1@{k_final}": float("nan"),
            "rr": float("nan"), "ap": float("nan"),
            f"ndcg@{k_final}": float("nan"),
        }

        t_retr.append(t1 - t0)
        t_all.append(t1 - t0)
        rows.append({
            "question": q,
            "strategy": strategy,
            "k": k_final,
            "reranker": "ON" if use_reranker else "OFF",
            "mmr_lambda": mmr_lambda if "MMR" in strategy else None,
            "mmr_fetch_k": mmr_fetch_k if "MMR" in strategy or "Keyword" in strategy else None,
            "parent_merge_chars": parent_merge_chars if "Page-Group" in strategy else None,
            "num_expansions": num_expansions if "Multi-Query" in strategy else None,
            "must_keywords": must_keywords if "Keyword" in strategy else None,
            f"retrieved_sources@{k_final}": "; ".join(top_raw),
            "retrieval_time_s": t1 - t0,
            # di pipeline ini e2e == retrieval (jawaban tdk dihitung)
            "end2end_time_s": t1 - t0,
            **metr,
        })

    detail = pd.DataFrame(rows)
    summary = {
        "strategy": strategy,
        "reranker": "ON" if use_reranker else "OFF",
        "k": k_final,
        "mmr_lambda": mmr_lambda if "MMR" in strategy else None,
        "mmr_fetch_k": mmr_fetch_k if "MMR" in strategy or "Keyword" in strategy else None,
        "parent_merge_chars": parent_merge_chars if "Page-Group" in strategy else None,
        "num_expansions": num_expansions if "Multi-Query" in strategy else None,
        "must_keywords": must_keywords if "Keyword" in strategy else None,
        f"hits@{k_final}_avg": float(np.nanmean(detail.get(f"hits@{k_final}", np.nan))) if len(detail) else np.nan,
        f"precision@{k_final}_avg": float(np.nanmean(detail.get(f"precision@{k_final}", np.nan))) if len(detail) else np.nan,
        f"recall@{k_final}_avg": float(np.nanmean(detail.get(f"recall@{k_final}", np.nan))) if len(detail) else np.nan,
        f"f1@{k_final}_avg": float(np.nanmean(detail.get(f"f1@{k_final}", np.nan))) if len(detail) else np.nan,
        "mrr": float(np.nanmean(detail.get("rr", np.nan))) if len(detail) else np.nan,
        "map": float(np.nanmean(detail.get("ap", np.nan))) if len(detail) else np.nan,
        f"ndcg@{k_final}_avg": float(np.nanmean(detail.get(f"ndcg@{k_final}", np.nan))) if len(detail) else np.nan,
        "retrieval_time_avg_s": float(np.nanmean(detail.get("retrieval_time_s", np.nan))) if len(detail) else np.nan,
        "end2end_time_avg_s": float(np.nanmean(detail.get("end2end_time_s", np.nan))) if len(detail) else np.nan,
        "n_samples": int(len(detail)),
    }
    return detail, summary

# ====== Orkestrator Ablation ======


def run_ablation(
    df_eval: pd.DataFrame,
    index: BM25Index,
    *,
    setup: AblationSetup,
    llm: Optional[Any] = None,   # pass LLM jika memakai Multi-Query/HyDE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if setup.col_question not in df_eval.columns:
        raise ValueError(f"CSV must contain column '{setup.col_question}'.")

    all_details: List[pd.DataFrame] = []
    summaries: List[Dict[str, Any]] = []

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

    for (strategy, k_final, rr_flag, mmr_lam, mmr_fk, parent_chars, n_exp, must_kw) in grid:
        use_rr = (str(rr_flag).upper() == "ON")

        detail, summary = _run_single_config(
            df_eval, index,
            strategy=strategy,
            k_final=int(k_final),
            use_reranker=use_rr,
            llm=llm,
            mmr_lambda=float(mmr_lam),
            mmr_fetch_k=int(mmr_fk),
            parent_merge_chars=int(parent_chars),
            num_expansions=int(n_exp),
            must_keywords=str(must_kw or ""),
            col_question=setup.col_question,
            col_gold_label=setup.col_gold_label,
            col_gold_mode=setup.col_gold_mode,
            normalize_labels=setup.normalize_labels,
            include_page_suffix=setup.include_page_suffix,
        )
        if not detail.empty:
            all_details.append(detail)
        summaries.append(summary)

    detail_all = pd.concat(
        all_details, ignore_index=True) if all_details else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)

    # Simpan CSV
    if setup.save_dir:
        outdir = Path(setup.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            summary_df.to_csv(
                outdir / "ablation_summary_full.csv", index=False)
        except Exception:
            pass
        if not detail_all.empty:
            try:
                detail_all.to_csv(outdir / "ablation_details.csv", index=False)
            except Exception:
                pass

    return detail_all, summary_df


__all__ = ["AblationSetup", "run_ablation"]
