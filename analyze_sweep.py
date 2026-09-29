"""analyze_sweep.py - analisis lanjutan hasil sweep.

Menghasilkan tiga keluaran yang tidak ada di ringkasan sweep:

1. tabel_utama.csv      - Tabel 1 siap pakai: 5 setup x K x 7 metrik.
2. per_kategori.csv     - rincian per question_type dan difficulty, untuk
                          menguji klaim seperti "HyDE unggul pada pertanyaan
                          semantik" alih-alih mengandalkan rata-rata global.
3. signifikansi.csv     - uji berpasangan tiap setup melawan baseline BM25.

Dipakai Wilcoxon signed-rank, bukan uji-t: nilai RR dan nDCG per pertanyaan
menumpuk di 0 dan 1 sehingga asumsi kenormalan uji-t tidak terpenuhi.
Dilaporkan juga ukuran efek (rank-biserial) karena p-value pada n=350 mudah
menjadi signifikan walau selisihnya kecil.

Jalankan: python3 analyze_sweep.py [--dir DIR] [--granularitas dokumen|halaman]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluasi-set-507.csv"
DEFAULT_DIR = HERE / "Result" / "2026-09-13_sweep-350"

URUTAN = ["BM25", "BM25+Rerank", "BM25+MultiQuery", "BM25+HyDE",
          "BM25+MultiQuery+Rerank", "BM25+HyDE+Rerank"]
BASELINE = "BM25"


def muat_detail(folder: Path, gran: str) -> pd.DataFrame:
    """Gabungkan semua detail_*.csv menjadi satu tabel panjang."""
    pola = re.compile(rf"detail_{gran}_(.+)_k(\d+)\.csv$")
    bagian = []
    for f in sorted(folder.glob(f"detail_{gran}_*.csv")):
        m = pola.search(f.name)
        if not m:
            continue
        setup, k = m.group(1), int(m.group(2))
        d = pd.read_csv(f)
        # samakan nama kolom ber-@k supaya bisa digabung antar nilai k
        d = d.rename(columns={f"hits@{k}": "hit", f"precision@{k}": "precision",
                              f"recall@{k}": "recall", f"f1@{k}": "f1",
                              f"ndcg@{k}": "ndcg", "rr": "rr", "ap": "ap"})
        d["setup"], d["k"] = setup, k
        bagian.append(d)
    if not bagian:
        raise SystemExit(f"[FATAL] tidak ada detail_{gran}_*.csv di {folder}")
    return pd.concat(bagian, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--granularitas", default="dokumen", choices=["dokumen", "halaman"])
    args = ap.parse_args()
    folder, gran = Path(args.dir), args.granularitas

    from scipy.stats import wilcoxon

    meta = pd.read_csv(CSV, sep=";")[
        ["question", "question_type", "difficulty", "answerable", "gold_mode"]]
    det = muat_detail(folder, gran).merge(meta, on="question", how="left")

    hilang = int(det["question_type"].isna().sum())
    if hilang:
        print(f"[PERINGATAN] {hilang} baris detail tidak cocok dengan dataset")

    METRIK = ["hit", "precision", "recall", "f1", "rr", "ap", "ndcg"]
    NAMA = {"hit": "HIT@K", "precision": "precision@K", "recall": "recall@K",
            "f1": "f1@K", "rr": "MRR", "ap": "MAP", "ndcg": "NDCG@K"}

    # ---------- 1. Tabel utama ----------
    jawabable = det[det["answerable"] == "yes"]
    utama = (jawabable.groupby(["k", "setup"])[METRIK]
             .agg(["mean", "size"]).reset_index())
    utama.columns = ["k", "setup"] + [f"{a}|{b}" for a, b in utama.columns[2:]]
    utama["n"] = utama[f"{METRIK[0]}|size"].astype(int)
    utama = utama[["k", "setup", "n"] + [f"{m}|mean" for m in METRIK]]
    utama.columns = ["k", "setup", "n"] + [NAMA[m] for m in METRIK]
    utama["setup"] = pd.Categorical(utama["setup"], URUTAN, ordered=True)
    utama = utama.sort_values(["k", "setup"])
    utama.to_csv(folder / f"tabel_utama_{gran}.csv", index=False)

    print(f"=== TABEL UTAMA ({gran}, n={jawabable['question'].nunique()} pertanyaan) ===")
    for k in sorted(utama["k"].unique()):
        print(f"\nK={k}")
        sub = utama[utama["k"] == k]
        print(f"  {'SETUP':24s} {'HIT@K':>7s} {'MRR':>7s} {'MAP':>7s} {'NDCG@K':>7s} "
              f"{'prec@K':>7s} {'rec@K':>7s} {'f1@K':>7s}")
        for _, r in sub.iterrows():
            print(f"  {str(r['setup']):24s} {r['HIT@K']:7.4f} {r['MRR']:7.4f} "
                  f"{r['MAP']:7.4f} {r['NDCG@K']:7.4f} {r['precision@K']:7.4f} "
                  f"{r['recall@K']:7.4f} {r['f1@K']:7.4f}")

    # ---------- 1b. Plafon precision ----------
    # precision@K = tp / (jumlah unit UNIK yang terambil), bukan tp/K, karena
    # rank_metrics mendeduplikasi kandidat. Plafonnya 1/n_unik untuk pertanyaan
    # bergold tunggal - perlu dilaporkan agar tabel tidak salah dibaca.
    print(f"\n\nPlafon precision@K (n_unik = rata-rata unit berbeda yang terambil):")
    for k in sorted(jawabable["k"].unique()):
        sub = jawabable[(jawabable["k"] == k) & (jawabable["setup"] == BASELINE)]
        if sub.empty or sub["precision"].eq(0).all():
            continue
        bukan_nol = sub[sub["precision"] > 0]
        n_unik = (bukan_nol["hit"] / bukan_nol["precision"]).mean()
        print(f"  K={k:<3d} n_unik rata-rata = {n_unik:5.2f}  "
              f"-> plafon precision = {1/n_unik:.3f}  "
              f"(teramati {sub['precision'].mean():.3f})")

    # ---------- 2. Rincian per kategori ----------
    baris = []
    for kolom in ("question_type", "difficulty"):
        g = (jawabable.groupby(["k", "setup", kolom])[METRIK].mean().reset_index())
        g = g.rename(columns={kolom: "kategori"})
        g.insert(0, "dimensi", kolom)
        baris.append(g)
    per_kat = pd.concat(baris, ignore_index=True)
    per_kat["n"] = per_kat.apply(lambda r: int(
        (jawabable[(jawabable["k"] == r["k"]) & (jawabable["setup"] == r["setup"]) &
                   (jawabable[r["dimensi"]] == r["kategori"])]).shape[0]), axis=1)
    per_kat.to_csv(folder / f"per_kategori_{gran}.csv", index=False)

    print(f"\n\n=== HYDE vs BM25 PER TIPE PERTANYAAN (K=10, {gran}) ===")
    p10 = per_kat[(per_kat["k"] == 10) & (per_kat["dimensi"] == "question_type")]
    piv = p10.pivot_table(index="kategori", columns="setup", values="ndcg")
    if BASELINE in piv.columns and "BM25+HyDE" in piv.columns:
        piv["selisih_nDCG"] = piv["BM25+HyDE"] - piv[BASELINE]
        n_per = p10.groupby("kategori")["n"].first()
        print(f"  {'kategori':14s} {'n':>4s} {'BM25':>8s} {'HyDE':>8s} {'selisih':>9s}")
        for kat in piv.sort_values("selisih_nDCG", ascending=False).index:
            print(f"  {kat:14s} {n_per[kat]:4d} {piv.loc[kat, BASELINE]:8.4f} "
                  f"{piv.loc[kat, 'BM25+HyDE']:8.4f} {piv.loc[kat, 'selisih_nDCG']:+9.4f}")

    # ---------- 3. Uji signifikansi berpasangan ----------
    hasil = []
    for k in sorted(det["k"].unique()):
        dasar = jawabable[(jawabable["k"] == k) & (jawabable["setup"] == BASELINE)]
        for setup in URUTAN:
            if setup == BASELINE:
                continue
            lawan = jawabable[(jawabable["k"] == k) & (jawabable["setup"] == setup)]
            gab = dasar.merge(lawan, on="question", suffixes=("_a", "_b"))
            if gab.empty:
                continue
            for m in ("rr", "ap", "ndcg", "hit"):
                a, b = gab[f"{m}_a"], gab[f"{m}_b"]
                beda = (b - a).dropna()
                n_beda = int((beda != 0).sum())
                if n_beda == 0:
                    hasil.append({"k": k, "setup": setup, "metrik": NAMA[m],
                                  "n_pasang": len(beda), "n_berbeda": 0,
                                  "median_selisih": 0.0, "rata_selisih": 0.0,
                                  "p_value": 1.0, "rank_biserial": 0.0,
                                  "signifikan_0.05": False})
                    continue
                stat, p = wilcoxon(b, a, zero_method="wilcox", alternative="two-sided")
                # rank-biserial: proporsi pasangan membaik dikurangi yang memburuk
                naik = int((beda > 0).sum())
                turun = int((beda < 0).sum())
                rb = (naik - turun) / n_beda
                hasil.append({"k": k, "setup": setup, "metrik": NAMA[m],
                              "n_pasang": len(beda), "n_berbeda": n_beda,
                              "median_selisih": float(beda[beda != 0].median()),
                              "rata_selisih": float(beda.mean()),
                              "p_value": float(p), "rank_biserial": round(rb, 4),
                              "signifikan_0.05": bool(p < 0.05)})
    sig = pd.DataFrame(hasil)
    if sig.empty:
        print("\n[INFO] belum ada setup pembanding selain baseline; "
              "uji signifikansi dilewati")
    else:
        sig.to_csv(folder / f"signifikansi_{gran}.csv", index=False)

    print(f"\n\n=== SIGNIFIKANSI vs BM25 (Wilcoxon, {gran}) ===")
    print("  rank-biserial = proporsi pertanyaan membaik dikurangi yang memburuk")
    for k in (sorted(sig["k"].unique()) if not sig.empty else []):
        print(f"\nK={k}")
        print(f"  {'setup':24s} {'metrik':8s} {'Δrata':>9s} {'p':>10s} {'efek':>7s}  ket")
        for _, r in sig[sig["k"] == k].iterrows():
            tanda = "signifikan" if r["signifikan_0.05"] else "tidak signifikan"
            print(f"  {r['setup']:24s} {r['metrik']:8s} {r['rata_selisih']:+9.4f} "
                  f"{r['p_value']:10.2e} {r['rank_biserial']:+7.3f}  {tanda}")

    # ---------- 4. Pertanyaan unanswerable ----------
    tak = det[det["answerable"] == "no"]["question"].nunique()
    print(f"\n\n=== PERTANYAAN UNANSWERABLE ===")
    print(f"  {tak} pertanyaan gold-nya kosong sehingga metrik retrieval-nya NaN")
    print("  dan TIDAK masuk rata-rata di tabel utama.")
    print("  Yang perlu diukur pada kelompok ini bukan retrieval melainkan apakah")
    print("  sistem menolak menjawab. Itu hanya bisa diukur pada tahap generasi,")
    print("  bukan dari sweep retrieval ini. Lihat evaluator.evaluate_df().")

    print(f"\nDisimpan di {folder}:")
    for nama in (f"tabel_utama_{gran}.csv", f"per_kategori_{gran}.csv",
                 f"signifikansi_{gran}.csv"):
        print(f"  {nama}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
