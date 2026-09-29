"""analisa_reranker.py - analisis lengkap perbandingan reranker.

Membaca checkpoint per model dari bandingkan_reranker2.py lalu menghasilkan:

  1. Tabel utama   - semua model, dua granularitas, K=5 dan K=10
  2. Kelas tokenizer - monolingual vs multilingual, diuji sebagai kelompok
  3. Biaya vs mutu - ms/pertanyaan terhadap perolehan MRR
  4. Per tipe pertanyaan untuk model terbaik dan model yang dipakai naskah
  5. Uji Wilcoxon + rank-biserial terhadap acuan tanpa rerank

Variabel yang dianalisis adalah CAKUPAN TOKENIZER, bukan label bahasa pada
kartu model. bge-reranker-large berlabel "en,zh" tetapi tulang punggungnya
XLM-RoBERTa berkosakata 250k, jadi ia memperlakukan teks Indonesia dengan
benar. Yang menentukan ukuran kosakata, dan itu dibaca langsung dari
config.json tiap model, bukan dari klaim kartunya.

Jalankan: python3 analisa_reranker.py [--k 10] [--gran halaman]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
CKPT = HERE / "Result" / "2026-09-13_reranker2" / "checkpoint"
OUT = HERE / "Result" / "2026-09-13_reranker2"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))

REPO = {
    "ms-marco-L6": "cross-encoder--ms-marco-MiniLM-L-6-v2",
    "ms-marco-L12": "cross-encoder--ms-marco-MiniLM-L-12-v2",
    "gte-mbert-en": "Alibaba-NLP--gte-reranker-modernbert-base",
    "mmarco-L6": "unicamp-dl--mMiniLM-L6-v2-mmarco-v2",
    "mmarco-L12": "cross-encoder--mmarco-mMiniLMv2-L12-H384-v1",
    "bge-base": "BAAI--bge-reranker-base",
    "bge-large-en": "BAAI--bge-reranker-large",
    "bge-v2-m3": "BAAI--bge-reranker-v2-m3",
    "gte-base": "Alibaba-NLP--gte-multilingual-reranker-base",
    "jina-v2": "jinaai--jina-reranker-v2-base-multilingual",
    "mxbai-v2": "mixedbread-ai--mxbai-rerank-base-v2",
}
# ms/pertanyaan, dari jalan.log (48 kandidat, CPU Apple Silicon)
MS = {"ms-marco-L6": 221, "mmarco-L6": 203, "bge-large-en": 2836,
      "bge-v2-m3": 3062, "ms-marco-L12": 678, "mmarco-L12": 400,
      "gte-base": 3494, "bge-base": 808, "gte-mbert-en": 4086,
      "jina-v2": 1187}


def konfig(kunci: str) -> dict:
    f = glob.glob(str(HUB / f"models--{REPO[kunci]}" / "snapshots" / "*" / "config.json"))
    if not f:
        return {}
    return json.load(open(f[0]))


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Ukuran efek berpasangan: proporsi pasangan membaik dikurangi memburuk."""
    d = b - a
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    return float((d > 0).mean() - (d < 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gran", default="halaman", choices=["halaman", "dokumen"])
    args = ap.parse_args()
    pre = f"{args.gran}_k{args.k}"

    f_ac = CKPT / "tanpa_rerank.csv"
    if not f_ac.exists():
        raise SystemExit(f"[FATAL] {f_ac} tidak ada")
    ac = pd.read_csv(f_ac).set_index("id")

    data, meta = {"tanpa_rerank": ac}, {}
    for kunci in REPO:
        f = CKPT / f"{kunci}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("id")
        if not d.index.equals(ac.index):
            print(f"[LEWAT] {kunci}: himpunan pertanyaan berbeda dari acuan")
            continue
        data[kunci] = d
        c = konfig(kunci)
        v = int(c.get("vocab_size", 0))
        meta[kunci] = {"kosakata": v,
                       "kelas": "monolingual" if v < 100_000 else "multilingual",
                       "tulang_punggung": c.get("model_type", "?"),
                       "lapis": c.get("num_hidden_layers", "?"),
                       "dim": c.get("hidden_size", "?"),
                       "ms": MS.get(kunci, np.nan)}

    n = len(ac)
    print(f"\n{'='*104}")
    print(f"PERBANDINGAN RERANKER  -  {len(meta)} model + acuan")
    print(f"dataset evaluasi 507 pertanyaan, {n} berjawab (yang dianalisis)")
    print(f"granularitas {args.gran}, K={args.k}, kandidat top-48 BM25")
    print(f"{'='*104}\n")

    # ---------------- 1. tabel utama ----------------
    baris = []
    for kunci, d in data.items():
        b = {"model": kunci,
             "HIT": d[f"{pre}_hit"].mean(),
             "MRR": d[f"{pre}_rr"].mean(),
             "nDCG": d[f"{pre}_ndcg"].mean()}
        b.update(meta.get(kunci, {}))
        if kunci != "tanpa_rerank":
            for met, kol in (("rr", "MRR"), ("ndcg", "nDCG")):
                a_, c_ = ac[f"{pre}_{met}"].values, d[f"{pre}_{met}"].values
                b[f"d{kol}"] = c_.mean() - a_.mean()
                b[f"p{kol}"] = (wilcoxon(c_, a_, zero_method="wilcox")[1]
                                if (c_ - a_).any() else np.nan)
                b[f"ef{kol}"] = rank_biserial(a_, c_)
        baris.append(b)
    t = pd.DataFrame(baris).sort_values("MRR", ascending=False)

    print(f"{'model':14s}{'tokenizer':13s}{'kosakata':>9s}{'lapis':>6s}"
          f"{'ms/q':>7s}{'HIT':>8s}{'MRR':>8s}{'nDCG':>8s}{'dMRR':>9s}"
          f"{'p':>10s}{'efek':>7s}")
    print("-" * 104)
    for _, x in t.iterrows():
        if x.model == "tanpa_rerank":
            print(f"{'tanpa rerank':14s}{'-':13s}{'-':>9s}{'-':>6s}{'-':>7s}"
                  f"{x.HIT:8.4f}{x.MRR:8.4f}{x.nDCG:8.4f}{'  (acuan)':>26s}")
            continue
        kls = "mono" if x.kelas == "monolingual" else "multi"
        tanda = "*" if x.pMRR < 0.05 else " "
        print(f"{x.model:14s}{kls:13s}{x.kosakata:>9,}{x.lapis:>6}"
              f"{x.ms:>7.0f}{x.HIT:8.4f}{x.MRR:8.4f}{x.nDCG:8.4f}"
              f"{x.dMRR:+9.4f}{x.pMRR:10.1e}{tanda}{x.efMRR:+6.2f}")
    print("\n* = beda signifikan terhadap tanpa-rerank (Wilcoxon berpasangan, p<0,05)")
    print("efek = rank-biserial: proporsi pertanyaan membaik dikurangi memburuk")

    # ---------------- 2. kelas tokenizer ----------------
    print(f"\n{'='*104}\nKELAS TOKENIZER\n")
    mod = t[t.model != "tanpa_rerank"]
    for kls in ("monolingual", "multilingual"):
        s = mod[mod.kelas == kls]
        if s.empty:
            continue
        untung = (s.dMRR > 0).sum()
        print(f"  {kls:13s} n={len(s)} model | dMRR dari {s.dMRR.min():+.4f} "
              f"sampai {s.dMRR.max():+.4f} | membantu {untung}/{len(s)}")
        print(f"                  {', '.join(s.model)}")
    mono, multi = mod[mod.kelas == "monolingual"], mod[mod.kelas == "multilingual"]
    if not mono.empty and not multi.empty:
        print(f"\n  Pemisahan: model multilingual TERBURUK ({multi.dMRR.min():+.4f}) "
              f"vs monolingual TERBAIK ({mono.dMRR.max():+.4f})")
        print(f"  {'TIDAK ADA TUMPANG TINDIH' if multi.dMRR.min() > mono.dMRR.max() else 'ADA TUMPANG TINDIH'}"
              f" antara kedua kelas.")

    # ---------------- 3. biaya vs mutu ----------------
    print(f"\n{'='*104}\nBIAYA vs MUTU (hanya model yang membantu)\n")
    s = mod[(mod.dMRR > 0) & mod.ms.notna()].sort_values("ms")
    print(f"  {'model':14s}{'ms/q':>7s}{'dMRR':>9s}{'dMRR per detik':>17s}")
    for _, x in s.iterrows():
        print(f"  {x.model:14s}{x.ms:>7.0f}{x.dMRR:+9.4f}{x.dMRR/(x.ms/1000):>17.4f}")
    if not s.empty:
        best = s.loc[s.dMRR.idxmax()]
        eff = s.loc[(s.dMRR / s.ms).idxmax()]
        print(f"\n  MRR tertinggi   : {best.model} ({best.dMRR:+.4f}, {best.ms:.0f} ms)")
        print(f"  Paling efisien  : {eff.model} ({eff.dMRR:+.4f}, {eff.ms:.0f} ms) "
              f"-> {best.ms/eff.ms:.1f}x lebih cepat, "
              f"selisih MRR hanya {best.dMRR-eff.dMRR:.4f}")

    # ---------------- 4. per tipe pertanyaan ----------------
    print(f"\n{'='*104}\nPER TIPE PERTANYAAN (nDCG@{args.k})\n")
    urut = mod.sort_values("MRR", ascending=False)
    pilih = [urut.iloc[0].model, "ms-marco-L6"]
    pilih = [p for p in dict.fromkeys(pilih) if p in data]
    tipe = ac["question_type"]
    print(f"  {'tipe':12s}{'n':>5s}{'acuan':>9s}" +
          "".join(f"{p:>15s}" for p in pilih))
    for tp in tipe.value_counts().index:
        m = tipe == tp
        dasar = ac.loc[m, f"{pre}_ndcg"].mean()
        sel = "".join(f"{data[p].loc[m, f'{pre}_ndcg'].mean()-dasar:>+15.4f}"
                      for p in pilih)
        print(f"  {tp:12s}{int(m.sum()):>5d}{dasar:>9.4f}{sel}")
    print(f"\n  (angka di bawah nama model = selisih terhadap acuan tanpa rerank)")

    t.to_csv(OUT / f"analisa_{args.gran}_k{args.k}.csv", index=False)
    print(f"\nTersimpan: {OUT / f'analisa_{args.gran}_k{args.k}.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
