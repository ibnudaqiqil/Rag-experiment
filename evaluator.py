# evaluator.py
# ---------------------------------------------------
# Jawaban berbasis konteks + evaluasi retrieval & jawaban
# ---------------------------------------------------

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import os
import numpy as np
import pandas as pd
from langchain.schema import HumanMessage, SystemMessage, Document

import llm_cache
from retrieval import retrieve
from utils import (
    rank_metrics, parse_gold_set, normalize_tag, normalize_tag_list,
    compute_answer_match, cosine_similarity_answer
)

# ---------- LLM factory ----------


def ensure_llm(provider: str, model: str, temperature: float = 0.2) -> Optional[Any]:
    """
    Buat Chat LLM sesuai provider. Kembalikan None jika 'Non-LLM'.
    """
    if provider in ("Non-LLM", "None", "", None):
        return None

    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        api = os.getenv("OPENAI_API_KEY", "")
        if not api:
            raise RuntimeError("OPENAI_API_KEY belum diset.")
        return ChatOpenAI(model=model, temperature=temperature)

    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api = os.getenv("GOOGLE_API_KEY", "")
        if not api:
            raise RuntimeError("GOOGLE_API_KEY belum diset.")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    if provider == "DeepSeek":
        from langchain_openai import ChatOpenAI
        api = os.getenv("DEEPSEEK_API_KEY", "")
        if not api:
            raise RuntimeError("DEEPSEEK_API_KEY belum diset.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api, base_url="https://api.deepseek.com")

    if provider == "OpenRouter (OpenAI-compatible)":
        from langchain_openai import ChatOpenAI
        api = os.getenv("OPENROUTER_API_KEY", "")
        if not api:
            raise RuntimeError("OPENROUTER_API_KEY belum diset.")
        return ChatOpenAI(model=model, temperature=temperature, api_key=api, base_url="https://openrouter.ai/api/v1")

    # default fallback
    return None


# ---------- QA generator ----------
def generate_answer(llm: Optional[Any], query: str, docs: List[Document]) -> Tuple[str, List[str]]:
    """
    Susun jawaban dari konteks terpilih. Jika llm None → tampilkan ringkasan konteks.
    """
    context_blocks, tags = [], []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "doc")
        page = meta.get("page", None)
        tag = f"{src}" + (f":p{page}" if page is not None else "")
        tags.append(tag)
        context_blocks.append(f"[{i}] {d.page_content}")

    if llm is None:
        text = "(Non-LLM) Top contexts:\n\n" + \
            "\n\n---\n\n".join(context_blocks[:3])
        return text, tags

    system = SystemMessage(content=(
        "You are a helpful assistant that answers strictly based on the provided context. "
        "Use inline citations like [1], [2] that refer to passage indices; if insufficient, say so."
    ))
    prompt = (
        f"User query: {query}\n\nContext passages:\n" +
        ("\n\n".join(context_blocks[:8]) if context_blocks else "No context.") +
        "\n\nInstructions:\n- Answer ONLY from the context.\n- Cite using [#] indices.\n- Be concise."
    )
    # Kunci cache memakai prompt lengkap, bukan query saja: jawaban bergantung
    # pada konteks yang terambil, sehingga konteks berbeda harus jadi entri
    # berbeda. Semua jawaban tersimpan di llm_cache/responses.jsonl.
    provider = type(llm).__name__
    model = str(getattr(llm, "model_name", None) or getattr(llm, "model", "?"))
    content = llm_cache.ambil_atau_panggil(
        "generate_answer", provider, model, f"{query}\n<<<CTX>>>\n{prompt}", 1, prompt,
        lambda: getattr(llm([system, HumanMessage(content=prompt)]), "content", ""))
    if content is None:
        return "(gagal memanggil LLM)", tags
    return content.strip(), tags


# ---------- Evaluasi per DataFrame ----------
def evaluate_df(
    df_eval: pd.DataFrame,
    index,
    # RetrievalParams dari retrieval.py (atau object serupa)
    params,
    *, llm: Optional[Any],
    reranker: Optional[Any],
    k_eval: int,
    cosine_thr: float,
    f1_thr: float,
    use_em: bool,
    col_question: str = "question",
    col_ref_answer: str = "reference_answer",
    col_gold_label: str = "gold_label",
    col_gold_mode: str = "gold_mode",
    # normalisasi label: abaikan :pN secara default
    include_page_suffix: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Jalankan evaluasi end-to-end: retrieval -> jawaban -> metrik.
    CSV minimal: col 'question'; opsional 'reference_answer' & 'gold_label'.
    gold_label gunakan 'NamaFile[:pN]' dipisah '|'.
    """
    rows = []
    t_retr = []
    t_e2e = []

    for _, row in df_eval.iterrows():
        q = str(row.get(col_question, "")).strip()

        # Optional fields
        ref_ans = ""
        if col_ref_answer in df_eval.columns and not pd.isna(row.get(col_ref_answer, "")):
            ref_ans = str(row.get(col_ref_answer, "")).strip()

        gold = ""
        if col_gold_label in df_eval.columns and not pd.isna(row.get(col_gold_label, "")):
            gold = str(row.get(col_gold_label, "")).strip()

        # 'any' = entri gold adalah lokasi alternatif, cukup ketemu salah satu.
        gold_mode = "all"
        if col_gold_mode in df_eval.columns and not pd.isna(row.get(col_gold_mode, "")):
            gold_mode = str(row.get(col_gold_mode, "all")).strip() or "all"

        import time
        t0 = time.perf_counter()
        docs, _ = retrieve(q, index, llm=None, k_docs=params.k_final, use_reranker=(
            reranker is not None), return_debug=False)
        t1 = time.perf_counter()
        ans, _ = generate_answer(llm, q, docs)
        t2 = time.perf_counter()

        # top-k tags (normalized)
        top_sources_raw = []
        for d in docs[:k_eval]:
            m = d.metadata or {}
            src = str(m.get("source", ""))
            page = m.get("page", None)
            tag = f"{src}:p{page}" if (page is not None) else src
            top_sources_raw.append(tag)

        top_norm = normalize_tag_list(
            top_sources_raw, include_page_suffix=include_page_suffix)

        gold_set_raw = parse_gold_set(gold) if gold else set()
        gold_norm = {normalize_tag(
            g, include_page_suffix=include_page_suffix) for g in gold_set_raw}

        # Rank metrics
        if gold_norm:
            metr = rank_metrics(top_norm, gold_norm, k=k_eval, gold_mode=gold_mode)
        else:
            # Jika tak ada gold -> metrik retrieval jadi NaN (kecuali hits=0)
            metr = {
                f"hits@{k_eval}": 0.0,
                f"precision@{k_eval}": float("nan"),
                f"recall@{k_eval}": float("nan"),
                f"f1@{k_eval}": float("nan"),
                "rr": float("nan"),
                "ap": float("nan"),
                f"ndcg@{k_eval}": float("nan"),
            }

        # Answer metrics
        cos = cosine_similarity_answer(
            ans, ref_ans) if ref_ans else float("nan")
        ans_metrics = {"em": float("nan"), "f1": float(
            "nan"), "correct": float("nan")}
        if ref_ans:
            ans_metrics = compute_answer_match(
                ans, ref_ans, cos_sim=cos, use_em=use_em, cosine_thr=cosine_thr, f1_thr=f1_thr)

        t_retr.append(t1 - t0)
        t_e2e.append(t2 - t0)

        out = {
            "question": q,
            f"retrieved_sources@{k_eval}": "; ".join(top_sources_raw),
            "cosine_sim_answer_ref": cos,
            "retrieval_time_s": t1 - t0,
            "end2end_time_s": t2 - t0,
        }
        out.update(metr)
        out.update(
            {"answer_em": ans_metrics["em"], "answer_f1": ans_metrics["f1"], "answer_correct": ans_metrics["correct"]})
        rows.append(out)

    detail = pd.DataFrame(rows)
    summary = {
        "n_samples": int(len(detail)),
        f"hits@{k_eval}_avg": float(np.nanmean(detail.get(f"hits@{k_eval}", np.nan))) if len(detail) else np.nan,
        f"precision@{k_eval}_avg": float(np.nanmean(detail.get(f"precision@{k_eval}", np.nan))) if len(detail) else np.nan,
        f"recall@{k_eval}_avg": float(np.nanmean(detail.get(f"recall@{k_eval}", np.nan))) if len(detail) else np.nan,
        f"f1@{k_eval}_avg": float(np.nanmean(detail.get(f"f1@{k_eval}", np.nan))) if len(detail) else np.nan,
        "mrr": float(np.nanmean(detail.get("rr", np.nan))) if len(detail) else np.nan,
        "map": float(np.nanmean(detail.get("ap", np.nan))) if len(detail) else np.nan,
        f"ndcg@{k_eval}_avg": float(np.nanmean(detail.get(f"ndcg@{k_eval}", np.nan))) if len(detail) else np.nan,
        "cosine_sim_avg": float(np.nanmean(detail.get("cosine_sim_answer_ref", np.nan))) if len(detail) else np.nan,
        "answer_em_avg": float(np.nanmean(detail.get("answer_em", np.nan))) if len(detail) else np.nan,
        "answer_f1_avg": float(np.nanmean(detail.get("answer_f1", np.nan))) if len(detail) else np.nan,
        "answer_acc_semantic": float(np.nanmean(detail.get("answer_correct", np.nan))) if len(detail) else np.nan,
        "retrieval_time_avg_s": float(np.nanmean(detail.get("retrieval_time_s", np.nan))) if len(detail) else np.nan,
        "end2end_time_avg_s": float(np.nanmean(detail.get("end2end_time_s", np.nan))) if len(detail) else np.nan,
    }
    return detail, summary


__all__ = ["ensure_llm", "generate_answer", "evaluate_df"]
