"""bandingkan_reranker.py - menguji apakah reranker Inggris penyebab penurunan.

Reranker default (cross-encoder/ms-marco-MiniLM-L-6-v2) dilatih pada MS MARCO
berbahasa Inggris, sedangkan korpus dan query di sini bahasa Indonesia.
Diagnostik sebelumnya menunjukkan reranker itu MENURUNKAN MRR: 0,069 pada
level dokumen dan 0,110 pada level halaman.

Skrip ini membandingkan tiga kondisi pada pertanyaan yang sama:
  1. tanpa rerank (urutan BM25 apa adanya)
  2. reranker Inggris  (ms-marco-MiniLM-L-6-v2)
  3. reranker multilingual (default BAAI/bge-reranker-v2-m3)

Kalau reranker multilingual memperbaiki hasil, ketidakcocokan bahasa terbukti
sebagai penyebab. Kalau sama-sama memburuk, klaim reranking harus dicabut,
bukan sekadar diganti modelnya.

Jalankan: python3 bandingkan_reranker.py [--k 10] [--n 120] [--fetch 40]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bm25_index import BM25Index  # noqa: E402
from utils import normalize_tag_list, parse_gold_set, normalize_tag, rank_metrics  # noqa: E402

CSV = HERE / "evaluasi-set-350.csv"
INDEX = HERE / "rag_cache" / "corpus18" / "bm25_index.pkl"
OUTDIR = HERE / "Result" / "2026-09-13_reranker"

INGGRIS = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MULTI = "BAAI/bge-reranker-v2-m3"


def tag(d) -> str:
    md = d.metadata or {}
    return f"{md.get('source', '')}:p{md.get('page')}"


def skor_model(model, query, docs, max_chars=1024):
    return model.predict([(query, (d.page_content or "")[:max_chars]) for d in docs],
                         show_progress_bar=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=120, help="jumlah pertanyaan yang diuji")
    ap.add_argument("--fetch", type=int, default=40, help="kandidat BM25 sebelum rerank")
    ap.add_argument("--multi", default=MULTI)
    ap.add_argument("--page-level", action="store_true", default=True)
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import CrossEncoder

    index = BM25Index.load(INDEX)
    if index is None:
        raise SystemExit(f"[FATAL] gagal memuat index {INDEX}")

    df = pd.read_csv(CSV, sep=";")
    df = df[df["answerable"] == "yes"]
    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=42)

    print(f"pertanyaan diuji : {len(df)}")
    print(f"kandidat BM25    : {args.fetch}  ->  top-{args.k} setelah rerank")
    print(f"reranker Inggris : {INGGRIS}")
    print(f"reranker multi   : {args.multi}\n", flush=True)

    t = time.perf_counter()
    m_en = CrossEncoder(INGGRIS)
    print(f"  model Inggris dimuat ({time.perf_counter()-t:.1f}s)", flush=True)
    t = time.perf_counter()
    m_ml = CrossEncoder(args.multi)
    print(f"  model multilingual dimuat ({time.perf_counter()-t:.1f}s)\n", flush=True)

    baris = []
    t0 = time.perf_counter()
    for i, r in enumerate(df.itertuples(index=False), start=1):
        q = str(r.question)
        gold = {normalize_tag(g, include_page_suffix=args.page_level)
                for g in parse_gold_set(str(r.gold_label))}
        mode = str(getattr(r, "gold_mode", "all") or "all")
        docs = [d for d, _ in index.search(q, topn=args.fetch)]
        if not docs:
            continue

        urutan = {
            "tanpa_rerank": docs,
            "rerank_inggris": [docs[j] for j in np.argsort(-skor_model(m_en, q, docs))],
            "rerank_multilingual": [docs[j] for j in np.argsort(-skor_model(m_ml, q, docs))],
        }
        catat = {"id": r.id, "question_type": r.question_type}
        for nama, seq in urutan.items():
            top = normalize_tag_list([tag(d) for d in seq[:args.k]],
                                     include_page_suffix=args.page_level)
            m = rank_metrics(top, gold, k=args.k, gold_mode=mode)
            catat[f"{nama}_hit"] = m[f"hits@{args.k}"]
            catat[f"{nama}_rr"] = m["rr"]
            catat[f"{nama}_ndcg"] = m[f"ndcg@{args.k}"]
        baris.append(catat)
        if i % 20 == 0:
            sisa = (time.perf_counter() - t0) / i * (len(df) - i)
            print(f"  {i}/{len(df)}  (perkiraan sisa {sisa/60:.1f} menit)", flush=True)

    hasil = pd.DataFrame(baris)
    out = OUTDIR / f"banding_reranker_k{args.k}.csv"
    hasil.to_csv(out, index=False)

    print(f"\n{'=' * 68}")
    print(f"HASIL (n={len(hasil)}, K={args.k}, gold level "
          f"{'halaman' if args.page_level else 'dokumen'})\n")
    print(f"  {'kondisi':22s} {'HIT@K':>8s} {'MRR':>8s} {'nDCG@K':>8s}")
    dasar = {}
    for nama in ("tanpa_rerank", "rerank_inggris", "rerank_multilingual"):
        h, rr, nd = (hasil[f"{nama}_hit"].mean(), hasil[f"{nama}_rr"].mean(),
                     hasil[f"{nama}_ndcg"].mean())
        if nama == "tanpa_rerank":
            dasar = {"hit": h, "rr": rr, "ndcg": nd}
            print(f"  {nama:22s} {h:8.4f} {rr:8.4f} {nd:8.4f}")
        else:
            print(f"  {nama:22s} {h:8.4f} {rr:8.4f} {nd:8.4f}   "
                  f"(Δ {h-dasar['hit']:+.4f} / {rr-dasar['rr']:+.4f} / "
                  f"{nd-dasar['ndcg']:+.4f})")

    from scipy.stats import wilcoxon
    print(f"\n  Uji Wilcoxon terhadap 'tanpa rerank':")
    for nama in ("rerank_inggris", "rerank_multilingual"):
        for m in ("rr", "ndcg"):
            a, b = hasil["tanpa_rerank_" + m], hasil[f"{nama}_{m}"]
            beda = (b - a)
            if (beda != 0).sum() == 0:
                print(f"    {nama:22s} {m:5s} tidak ada selisih")
                continue
            _, p = wilcoxon(b, a, zero_method="wilcox")
            arah = "MEMBAIK" if beda.mean() > 0 else "MEMBURUK"
            print(f"    {nama:22s} {m:5s} Δ={beda.mean():+.4f}  p={p:.2e}  "
                  f"{arah}{' (signifikan)' if p < 0.05 else ' (tidak signifikan)'}")

    print(f"\nwaktu: {(time.perf_counter()-t0)/60:.1f} menit")
    print(f"Rincian: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
