# rerun_ablation.py
# ---------------------------------------------------------------
# Menjalankan ulang ablation dengan rank_metrics() yang sudah diperbaiki,
# memakai input yang SAMA PERSIS dengan run sebelumnya supaya selisihnya
# murni berasal dari perbaikan metrik:
#   - dataset  : evaluasi-set.csv (199 baris, versi lama yang masih berduplikat)
#   - index    : rag_cache/default_corpus/bm25_index.pkl
#   - k        : 6
#   - reranker : ON dan OFF
#   - LLM      : tidak dipakai (retrieval saja)
#
# Hasil ditulis ke folder BARU. Result/ dan rag_cache/.../ablation/ yang lama
# tidak disentuh.
# ---------------------------------------------------------------

from __future__ import annotations
import os
import sys
from pathlib import Path

# Model reranker (cross-encoder/ms-marco-MiniLM-L-6-v2) sudah ada di cache
# HuggingFace lokal. Tanpa mode offline, Hub tetap melakukan permintaan
# jaringan untuk cek versi dan proses menggantung bila jaringan diblokir.
# Harus diset SEBELUM sentence_transformers/transformers diimpor.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd

from ablation import AblationSetup, run_ablation
from bm25_index import BM25Index

HERE = Path(__file__).resolve().parent
EVAL_CSV = HERE / "evaluasi-set.csv"
INDEX_PKL = HERE / "rag_cache" / "default_corpus" / "bm25_index.pkl"
OUT_DIR = HERE / "Result" / "2026-09-13_metrik-diperbaiki"

# Strategi tunggal: run sebelumnya hanya menghasilkan 2 baris (ON/OFF),
# jadi hanya satu strategi yang dipakai.
STRATEGY = "BM25 (standar)"
K_FINAL = 6


def main() -> int:
    if not INDEX_PKL.exists():
        print(f"[FATAL] index tidak ditemukan: {INDEX_PKL}")
        return 1

    index = BM25Index.load(INDEX_PKL)
    if index is None:
        print(f"[FATAL] gagal memuat index dari {INDEX_PKL}")
        return 1

    df = pd.read_csv(EVAL_CSV, sep=";")
    print(f"Dataset : {EVAL_CSV.name} ({len(df)} baris)")
    print(f"Index   : {len(index.docs)} chunk")
    print(f"Config  : strategi={STRATEGY!r} k={K_FINAL} reranker=ON/OFF LLM=tidak")
    print(f"Output  : {OUT_DIR}")

    setup = AblationSetup(
        strategies=(STRATEGY,),
        reranker_flags=("ON", "OFF"),
        k_list=(K_FINAL,),
        include_page_suffix=False,
        save_dir=str(OUT_DIR),
    )

    detail, summary = run_ablation(df, index, setup=setup, llm=None)

    cols = [c for c in summary.columns
            if c in ("reranker", f"hits@{K_FINAL}_avg", f"precision@{K_FINAL}_avg",
                     f"recall@{K_FINAL}_avg", f"f1@{K_FINAL}_avg", "mrr", "map",
                     f"ndcg@{K_FINAL}_avg", "n_samples")]
    print("\n=== HASIL SETELAH PERBAIKAN ===")
    print(summary[cols].to_string(index=False))

    # Sanity check: tidak boleh ada metrik di luar [0,1]
    bounded = [c for c in summary.columns
               if any(t in c.lower() for t in
                      ("precision", "recall", "f1", "ndcg", "mrr", "map", "hits"))]
    bad = []
    for c in bounded:
        s = pd.to_numeric(summary[c], errors="coerce")
        if (s > 1.0001).any():
            bad.append(f"{c} maks={s.max():.4f}")
    print("\nCek batas [0,1]:", "GAGAL -> " + ", ".join(bad) if bad else "OK, semua <= 1")

    dcols = [c for c in detail.columns
             if any(t in c.lower() for t in
                    ("precision", "recall", "f1", "ndcg", "rr", "ap", "hits"))]
    dbad = []
    for c in dcols:
        s = pd.to_numeric(detail[c], errors="coerce")
        if (s > 1.0001).any():
            dbad.append(f"{c} ({int((s > 1.0001).sum())} baris, maks {s.max():.4f})")
    print("Cek per-pertanyaan  :", "GAGAL -> " + ", ".join(dbad) if dbad else "OK, semua <= 1")
    return 1 if (bad or dbad) else 0


if __name__ == "__main__":
    sys.exit(main())
