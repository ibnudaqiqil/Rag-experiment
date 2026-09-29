"""banding_sweep.py - membandingkan dua run sweep pada pertanyaan yang sama.

Dipakai untuk menjawab satu pertanyaan: apakah mengganti model reranker
mengubah kesimpulan? Karena kedua run memakai dataset, index, dan ekspansi
LLM yang sama, satu-satunya yang berbeda adalah model rerankernya, sehingga
selisihnya bisa dibaca sebagai efek model itu.

Uji Wilcoxon signed-rank dipakai, bukan uji-t: nilai RR dan nDCG per
pertanyaan menumpuk di 0 dan 1 sehingga asumsi kenormalan tidak terpenuhi.
Ukuran efek rank-biserial dilaporkan karena pada n=471 p-value gampang
signifikan walau selisihnya kecil.

Jalankan: python3 banding_sweep.py --a DIR_LAMA --b DIR_BARU [--gran dokumen]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluasi-set-507.csv"
URUT = ["BM25", "BM25+Rerank", "BM25+MultiQuery", "BM25+HyDE",
        "BM25+MultiQuery+Rerank", "BM25+HyDE+Rerank"]


def muat(folder: Path, gran: str, jawabable: set[str]) -> dict:
    """{(setup, k): DataFrame} dengan kolom metrik yang sudah diseragamkan."""
    pola = re.compile(rf"detail_{gran}_(.+)_k(\d+)\.csv$")
    out = {}
    for f in sorted(folder.glob(f"detail_{gran}_*.csv")):
        m = pola.search(f.name)
        if not m:
            continue
        setup, k = m.group(1), int(m.group(2))
        d = pd.read_csv(f)
        d = d[d["question"].isin(jawabable)]
        d = d.rename(columns={f"hits@{k}": "hit", f"ndcg@{k}": "ndcg"})
        out[(setup, k)] = d.set_index("question")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run pembanding (mis. reranker lama)")
    ap.add_argument("--b", required=True, help="run baru")
    ap.add_argument("--gran", default="dokumen", choices=["dokumen", "halaman"])
    ap.add_argument("--csv", default=str(CSV))
    args = ap.parse_args()

    ev = list(csv.DictReader(open(args.csv, encoding="utf-8"), delimiter=";"))
    jawabable = {r["question"] for r in ev if r["answerable"] == "yes"}

    A = muat(Path(args.a), args.gran, jawabable)
    B = muat(Path(args.b), args.gran, jawabable)
    sama = sorted(set(A) & set(B), key=lambda x: (x[1], URUT.index(x[0])
                                                  if x[0] in URUT else 99))
    if not sama:
        print("[FATAL] tidak ada setup+k yang sama di kedua folder")
        return 1

    print(f"granularitas {args.gran}, n={len(jawabable)} pertanyaan answerable")
    print(f"  A = {Path(args.a).name}")
    print(f"  B = {Path(args.b).name}")
    kurang = sorted(set(A) - set(B))
    if kurang:
        print(f"  [INFO] {len(kurang)} sel belum ada di B: "
              + ", ".join(f"{s} k={k}" for s, k in kurang[:6]))
    print()

    from scipy.stats import wilcoxon

    METRIK = [("hit", "HIT@K"), ("rr", "MRR"), ("ap", "MAP"), ("ndcg", "NDCG@K")]
    print(f"  {'setup':24s} {'K':>3s} " + " ".join(f"{n:>17s}" for _, n in METRIK))
    baris_out = []
    for setup, k in sama:
        a, b = A[(setup, k)], B[(setup, k)]
        idx = a.index.intersection(b.index)
        sel = []
        for kol, nama in METRIK:
            x, y = a.loc[idx, kol], b.loc[idx, kol]
            d = (y - x).dropna()
            beda = int((d != 0).sum())
            p = wilcoxon(y, x, zero_method="wilcox")[1] if beda else 1.0
            bintang = "*" if p < 0.05 and beda else " "
            sel.append(f"{x.mean():.3f}->{y.mean():.3f}{bintang}")
            baris_out.append({"setup": setup, "k": k, "metrik": nama,
                              "A": round(x.mean(), 4), "B": round(y.mean(), 4),
                              "selisih": round(y.mean() - x.mean(), 4),
                              "n_berbeda": beda, "p_value": p,
                              "signifikan_0.05": bool(p < 0.05 and beda)})
        print(f"  {setup:24s} {k:3d} " + " ".join(f"{s:>17s}" for s in sel))

    out = Path(args.b) / f"banding_terhadap_{Path(args.a).name}_{args.gran}.csv"
    pd.DataFrame(baris_out).to_csv(out, index=False)
    print(f"\n  * = beda signifikan (Wilcoxon, p<0.05)")
    print(f"  tersimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
