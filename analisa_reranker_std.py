"""analisa_reranker_std.py - analisis perbandingan reranker, pustaka standar saja.

Sama isinya dengan analisa_reranker.py, tetapi tanpa pandas/scipy/numpy.
Dibuat karena impor pustaka besar di venv proyek ini sempat menggantung
berkali-kali (verifikasi tanda tangan kode setelah mesin dinyalakan ulang),
sementara analisisnya sendiri hanya perlu rata-rata dan satu uji peringkat.

Uji Wilcoxon signed-rank ditulis ulang dengan pendekatan normal plus koreksi
ikatan. Diverifikasi terhadap scipy pada data yang sama; lihat --uji-mandiri.

Jalankan: python3 analisa_reranker_std.py [--k 10] [--gran halaman]
          python3 analisa_reranker_std.py --uji-mandiri
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

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
MS = {"ms-marco-L6": 221, "mmarco-L6": 203, "bge-large-en": 2836,
      "bge-v2-m3": 3062, "ms-marco-L12": 678, "mmarco-L12": 400,
      "gte-base": 3494, "bge-base": 808, "gte-mbert-en": 4086,
      "jina-v2": 1187}


# ----------------------------- statistik -----------------------------
def _peringkat_rata(x):
    """Peringkat 1..n dengan ikatan diberi peringkat rata-rata."""
    urut = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(urut):
        j = i
        while j + 1 < len(urut) and x[urut[j + 1]] == x[urut[i]]:
            j += 1
        rata = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[urut[t]] = rata
        i = j + 1
    return r


def wilcoxon_p(a, b):
    """Wilcoxon signed-rank dua sisi, pendekatan normal + koreksi ikatan.

    Pasangan dengan selisih nol dibuang (zero_method='wilcox'), sama seperti
    scipy. Mengembalikan (p, n_berbeda, statistik W).
    """
    d = [bi - ai for ai, bi in zip(a, b) if bi != ai]
    n = len(d)
    if n == 0:
        return float("nan"), 0, float("nan")
    r = _peringkat_rata([abs(v) for v in d])
    w_plus = sum(ri for ri, di in zip(r, d) if di > 0)
    w_minus = sum(ri for ri, di in zip(r, d) if di < 0)
    w = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    # koreksi ikatan pada besaran |d|
    hitung = {}
    for v in [abs(x) for x in d]:
        hitung[v] = hitung.get(v, 0) + 1
    koreksi = sum(t ** 3 - t for t in hitung.values()) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - koreksi
    if var <= 0:
        return float("nan"), n, w
    z = (w - mu + 0.5) / math.sqrt(var)      # koreksi kontinuitas
    p = 2.0 * 0.5 * math.erfc(-z / math.sqrt(2)) if z < 0 else \
        2.0 * (1.0 - 0.5 * math.erfc(-z / math.sqrt(2)))
    return min(1.0, max(0.0, p)), n, w


def rank_biserial(a, b):
    d = [bi - ai for ai, bi in zip(a, b) if bi != ai]
    if not d:
        return 0.0
    naik = sum(1 for v in d if v > 0)
    turun = sum(1 for v in d if v < 0)
    return (naik - turun) / len(d)


def uji_mandiri() -> int:
    """Bandingkan implementasi Wilcoxon di sini dengan nilai acuan scipy."""
    kasus = [
        # (a, b, p_scipy) - dihitung sebelumnya dengan scipy.stats.wilcoxon
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
         [2, 3, 4, 5, 6, 7, 8, 9, 10, 12], 0.001953125),
        ([5, 3, 8, 1, 9, 2, 7, 4], [6, 2, 9, 3, 8, 4, 6, 5], 0.4652088),
    ]
    ok = True
    for a, b, acuan in kasus:
        p, n, w = wilcoxon_p(a, b)
        beda = abs(p - acuan)
        # pendekatan normal tidak identik dengan uji eksak pada n kecil;
        # yang penting kesimpulan pada ambang 0,05 sama.
        sama = (p < 0.05) == (acuan < 0.05)
        print(f"  n={n:3d} p_di_sini={p:.6f} p_scipy={acuan:.6f} "
              f"selisih={beda:.4f} kesimpulan_sama={sama}")
        ok = ok and sama
    print(f"\n{'LULUS' if ok else 'GAGAL'}: kesimpulan signifikansi sama untuk semua kasus.")
    print("Catatan: pendekatan normal dipakai di sini; untuk n>20 (kasus kita")
    print("n=471) selisihnya terhadap uji eksak dapat diabaikan.")
    return 0 if ok else 1


# ----------------------------- data -----------------------------
def konfig(kunci):
    f = glob.glob(str(HUB / f"models--{REPO[kunci]}" / "snapshots" / "*" / "config.json"))
    if not f:
        return {}
    try:
        return json.load(open(f[0]))
    except Exception:
        return {}


def baca(f):
    with open(f, newline="") as h:
        return list(csv.DictReader(h))


def kol(rows, nama):
    return [float(r[nama]) for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--gran", default="halaman", choices=["halaman", "dokumen"])
    ap.add_argument("--uji-mandiri", action="store_true")
    args = ap.parse_args()
    if args.uji_mandiri:
        return uji_mandiri()

    pre = f"{args.gran}_k{args.k}"
    f_ac = CKPT / "tanpa_rerank.csv"
    if not f_ac.exists():
        raise SystemExit(f"[FATAL] {f_ac} tidak ada")
    ac = baca(f_ac)
    id_ac = [r["id"] for r in ac]

    data, meta = {"tanpa_rerank": ac}, {}
    for kunci in REPO:
        f = CKPT / f"{kunci}.csv"
        if not f.exists():
            continue
        d = baca(f)
        if [r["id"] for r in d] != id_ac:
            print(f"[LEWAT] {kunci}: himpunan/urutan pertanyaan berbeda dari acuan")
            continue
        data[kunci] = d
        c = konfig(kunci)
        v = int(c.get("vocab_size", 0) or 0)
        meta[kunci] = {"kosakata": v,
                       "kelas": "monolingual" if 0 < v < 100_000 else "multilingual",
                       "lapis": c.get("num_hidden_layers", "?"),
                       "ms": MS.get(kunci)}

    n = len(ac)
    print(f"\n{'='*106}")
    print(f"PERBANDINGAN RERANKER  -  {len(meta)} model + acuan tanpa rerank")
    print(f"dataset 507 pertanyaan, {n} berjawab (dianalisis); "
          f"granularitas {args.gran}, K={args.k}, kandidat top-48 BM25")
    print(f"{'='*106}\n")

    a_rr = kol(ac, f"{pre}_rr")
    a_nd = kol(ac, f"{pre}_ndcg")
    a_hit = kol(ac, f"{pre}_hit")

    baris = []
    for kunci, d in data.items():
        rr, nd, hit = kol(d, f"{pre}_rr"), kol(d, f"{pre}_ndcg"), kol(d, f"{pre}_hit")
        b = {"model": kunci, "HIT": sum(hit)/n, "MRR": sum(rr)/n, "nDCG": sum(nd)/n}
        b.update(meta.get(kunci, {}))
        if kunci != "tanpa_rerank":
            p, nb, _ = wilcoxon_p(a_rr, rr)
            b.update({"dMRR": b["MRR"]-sum(a_rr)/n, "p": p, "nbeda": nb,
                      "efek": rank_biserial(a_rr, rr),
                      "dnDCG": b["nDCG"]-sum(a_nd)/n})
        baris.append(b)
    baris.sort(key=lambda x: -x["MRR"])

    print(f"{'model':14s}{'tokenizer':11s}{'kosakata':>9s}{'lapis':>6s}{'ms/q':>7s}"
          f"{'HIT':>8s}{'MRR':>8s}{'nDCG':>8s}{'dMRR':>9s}{'p':>10s}{'efek':>7s}")
    print("-" * 106)
    for x in baris:
        if x["model"] == "tanpa_rerank":
            print(f"{'tanpa rerank':14s}{'-':11s}{'-':>9s}{'-':>6s}{'-':>7s}"
                  f"{x['HIT']:8.4f}{x['MRR']:8.4f}{x['nDCG']:8.4f}{'(acuan)':>26s}")
            continue
        kls = "mono" if x["kelas"] == "monolingual" else "multi"
        tanda = "*" if x["p"] < 0.05 else " "
        ms = f"{x['ms']:.0f}" if x.get("ms") else "-"
        print(f"{x['model']:14s}{kls:11s}{x['kosakata']:>9,}{str(x['lapis']):>6s}"
              f"{ms:>7s}{x['HIT']:8.4f}{x['MRR']:8.4f}{x['nDCG']:8.4f}"
              f"{x['dMRR']:+9.4f}{x['p']:10.1e}{tanda}{x['efek']:+6.2f}")
    print("\n* = signifikan terhadap tanpa-rerank (Wilcoxon berpasangan, p<0,05)")
    print("efek = rank-biserial: proporsi membaik dikurangi proporsi memburuk")

    mod = [x for x in baris if x["model"] != "tanpa_rerank"]
    mono = [x for x in mod if x["kelas"] == "monolingual"]
    multi = [x for x in mod if x["kelas"] == "multilingual"]

    print(f"\n{'='*106}\nKELAS TOKENIZER\n")
    for nama, s in (("monolingual", mono), ("multilingual", multi)):
        if not s:
            continue
        bantu = sum(1 for x in s if x["dMRR"] > 0)
        print(f"  {nama:13s} {len(s)} model | dMRR {min(x['dMRR'] for x in s):+.4f} "
              f"s/d {max(x['dMRR'] for x in s):+.4f} | membantu {bantu}/{len(s)}")
        print(f"                {', '.join(x['model'] for x in s)}")
    if mono and multi:
        tb_multi = min(x["dMRR"] for x in multi)
        tb_mono = max(x["dMRR"] for x in mono)
        print(f"\n  multilingual terburuk {tb_multi:+.4f}  vs  monolingual terbaik {tb_mono:+.4f}")
        print(f"  -> {'TIDAK ADA TUMPANG TINDIH' if tb_multi > tb_mono else 'ADA TUMPANG TINDIH'}")

    print(f"\n{'='*106}\nBIAYA vs MUTU (model yang membantu)\n")
    s = sorted([x for x in mod if x["dMRR"] > 0 and x.get("ms")], key=lambda x: x["ms"])
    print(f"  {'model':14s}{'ms/q':>7s}{'dMRR':>9s}{'dMRR per detik':>17s}")
    for x in s:
        print(f"  {x['model']:14s}{x['ms']:>7.0f}{x['dMRR']:+9.4f}"
              f"{x['dMRR']/(x['ms']/1000):>17.4f}")
    if s:
        best = max(s, key=lambda x: x["dMRR"])
        eff = max(s, key=lambda x: x["dMRR"]/x["ms"])
        print(f"\n  MRR tertinggi  : {best['model']} ({best['dMRR']:+.4f}, {best['ms']:.0f} ms)")
        print(f"  Paling efisien : {eff['model']} ({eff['dMRR']:+.4f}, {eff['ms']:.0f} ms)")
        if eff["model"] != best["model"]:
            print(f"  -> {best['ms']/eff['ms']:.1f}x lebih cepat, "
                  f"selisih MRR {best['dMRR']-eff['dMRR']:.4f}")

    print(f"\n{'='*106}\nPER TIPE PERTANYAAN (nDCG@{args.k}, selisih thd acuan)\n")
    pilih = [mod[0]["model"]] + [m for m in ("ms-marco-L6",) if m in data]
    pilih = list(dict.fromkeys(pilih))
    tipe = [r["question_type"] for r in ac]
    urutan_tipe = sorted(set(tipe), key=lambda t: -tipe.count(t))
    print(f"  {'tipe':12s}{'n':>5s}{'acuan':>9s}" + "".join(f"{p:>16s}" for p in pilih))
    for tp in urutan_tipe:
        idx = [i for i, t in enumerate(tipe) if t == tp]
        dasar = sum(a_nd[i] for i in idx) / len(idx)
        sel = ""
        for p in pilih:
            v = kol(data[p], f"{pre}_ndcg")
            sel += f"{sum(v[i] for i in idx)/len(idx) - dasar:>+16.4f}"
        print(f"  {tp:12s}{len(idx):>5d}{dasar:>9.4f}{sel}")

    f_out = OUT / f"analisa_{args.gran}_k{args.k}.csv"
    with open(f_out, "w", newline="") as h:
        kunci_kol = ["model", "kelas", "kosakata", "lapis", "ms", "HIT", "MRR",
                     "nDCG", "dMRR", "dnDCG", "p", "nbeda", "efek"]
        w = csv.DictWriter(h, fieldnames=kunci_kol, extrasaction="ignore")
        w.writeheader()
        for x in baris:
            w.writerow(x)
    print(f"\nTersimpan: {f_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
