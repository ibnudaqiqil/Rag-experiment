"""format_tabel.py - cetak Tabel 1 dengan tata letak seperti di paper.

Format keluaran mengikuti tabel yang sudah ada: K dikelompokkan di kolom
kiri, 5 setup di bawahnya, nilai terbaik per kolom dalam tiap kelompok K
ditandai tebal.

Menghasilkan tiga bentuk sekaligus:
  - teks rata (untuk dibaca di terminal)
  - Markdown (untuk ditempel ke draf)
  - LaTeX booktabs (untuk naskah jurnal)

Jalankan: python3 format_tabel.py [--granularitas dokumen|halaman]
                                  [--metrik HIT@K MRR MAP NDCG@K ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "Result" / "2026-09-13_sweep-350"

URUTAN = ["BM25", "BM25 + Rerank", "BM25 + Multi-Query", "BM25 + HyDE",
          "BM25 + Multi-Query + Rerank", "BM25 + HyDE + Rerank"]
LABEL = {"BM25": "BM25", "BM25+Rerank": "BM25 + Rerank",
         "BM25+MultiQuery": "BM25 + Multi-Query",
         "BM25+HyDE": "BM25 + HyDE",
         "BM25+MultiQuery+Rerank": "BM25 + Multi-Query + Rerank",
         "BM25+HyDE+Rerank": "BM25 + HyDE + Rerank"}
# metrik yang "lebih besar lebih baik" - semuanya, tapi dibuat eksplisit
DEFAULT_METRIK = ["HIT@K", "MRR", "MAP", "NDCG@K"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--granularitas", default="dokumen", choices=["dokumen", "halaman"])
    ap.add_argument("--metrik", nargs="+", default=DEFAULT_METRIK)
    ap.add_argument("--desimal", type=int, default=3)
    args = ap.parse_args()

    f = Path(args.dir) / f"tabel_utama_{args.granularitas}.csv"
    if not f.exists():
        raise SystemExit(f"[FATAL] {f} belum ada. Jalankan analyze_sweep.py dulu.")
    df = pd.read_csv(f)

    kurang = [m for m in args.metrik if m not in df.columns]
    if kurang:
        raise SystemExit(f"[FATAL] kolom tidak ada: {kurang}\n"
                         f"        tersedia: {list(df.columns)}")

    df["label"] = df["setup"].map(LABEL).fillna(df["setup"])
    df["label"] = pd.Categorical(df["label"], URUTAN, ordered=True)
    df = df.sort_values(["k", "label"])
    d = args.desimal

    hilang = sorted(set(URUTAN) - set(df["label"].dropna().astype(str)))
    if hilang:
        print(f"[PERINGATAN] setup berikut TIDAK ADA di hasil, tabel belum lengkap:")
        for h in hilang:
            print(f"             {h}")
        print()

    lebar = max(len(s) for s in df["label"].astype(str)) + 2

    # ---------- teks ----------
    print(f"=== TABEL 1  (granularitas {args.granularitas}, "
          f"n={int(df['n'].iloc[0]) if 'n' in df.columns else '?'}) ===\n")
    head = f"{'K':<4s}{'SETUP':<{lebar}s}" + "".join(f"{m:>10s}" for m in args.metrik)
    print(head)
    print("-" * len(head))
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k]
        terbaik = {m: sub[m].max() for m in args.metrik}
        for i, (_, r) in enumerate(sub.iterrows()):
            kol = f"{k:<4d}" if i == 0 else " " * 4
            sel = ""
            for m in args.metrik:
                v = f"{r[m]:.{d}f}"
                sel += f"{('*' + v if abs(r[m] - terbaik[m]) < 1e-12 else v):>10s}"
            print(f"{kol}{str(r['label']):<{lebar}s}{sel}")
        print("-" * len(head))
    print("* = nilai terbaik pada kelompok K tersebut\n")

    # ---------- Markdown ----------
    print("\n=== MARKDOWN ===\n")
    print("| K | SETUP | " + " | ".join(args.metrik) + " |")
    print("|---|---|" + "---|" * len(args.metrik))
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k]
        terbaik = {m: sub[m].max() for m in args.metrik}
        for i, (_, r) in enumerate(sub.iterrows()):
            sel = []
            for m in args.metrik:
                v = f"{r[m]:.{d}f}"
                sel.append(f"**{v}**" if abs(r[m] - terbaik[m]) < 1e-12 else v)
            print(f"| {k if i == 0 else ''} | {r['label']} | " + " | ".join(sel) + " |")

    # ---------- LaTeX ----------
    print("\n\n=== LATEX ===\n")
    print("\\begin{tabular}{ll" + "r" * len(args.metrik) + "}")
    print("\\toprule")
    print("K & SETUP & " + " & ".join(m.replace("@", "@") for m in args.metrik) + " \\\\")
    print("\\midrule")
    for k in sorted(df["k"].unique()):
        sub = df[df["k"] == k]
        terbaik = {m: sub[m].max() for m in args.metrik}
        for i, (_, r) in enumerate(sub.iterrows()):
            sel = []
            for m in args.metrik:
                v = f"{r[m]:.{d}f}"
                sel.append(f"\\textbf{{{v}}}" if abs(r[m] - terbaik[m]) < 1e-12 else v)
            print(f"{k if i == 0 else ''} & {r['label']} & " + " & ".join(sel) + " \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
