"""bandingkan_reranker2.py - membandingkan banyak reranker, satu per satu.

Versi pertama (bandingkan_reranker.py) hanya menguji dua model dan punya satu
cacat: bge-reranker-v2-m3 berukuran 568M parameter sedangkan
ms-marco-MiniLM-L-6-v2 hanya 23M. Kalau yang multilingual menang, kita tidak
tahu penyebabnya bahasa atau sekadar model 25x lebih besar.

Skrip ini memisahkan kedua variabel itu dengan mengisi sumbu ukuran pada KEDUA
kelompok bahasa. Perbandingan yang paling menentukan ada dua:

  ms-marco-MiniLM-L-6-v2   23M   Inggris           MS MARCO
  mMiniLM-L6-v2-mmarco-v2 107M   multilingual      mMARCO

    Sama-sama MiniLM 6 lapis, data latih sama (mMARCO adalah MS MARCO yang
    diterjemahkan mesin, bahasa Indonesia termasuk). Selisihnya bahasa.

  bge-reranker-large      560M   Inggris/Mandarin  beragam
  bge-reranker-v2-m3      568M   multilingual      beragam

    Sama-sama BAAI, ukuran nyaris identik. Kalau yang pertama kalah, ukuran
    model terbukti bukan penjelasnya.

Acuan 'tanpa rerank' WAJIB ada. Tanpa itu kita hanya tahu model mana yang
terbaik di antara para reranker, bukan apakah me-rerank sama sekali membantu.

CATATAN MEMORI. Versi sebelumnya memuat semua model sekaligus (~5,6 GB bobot)
dan membuat mesin 16 GB kehabisan swap sampai proses hanya kebagian 5% CPU.
Di sini model dimuat SATU per SATU lalu dibebaskan, sehingga puncak memori
hanya sebesar model terbesar. Hasil tiap model langsung disimpan sebagai
checkpoint, jadi proses yang terputus bisa dilanjutkan tanpa mengulang.

Kandidat diambil 48 dokumen agar sama dengan pre_k pada run_sweep.py,
sehingga angkanya sebanding dengan Tabel 1.

Jalankan: python3 bandingkan_reranker2.py [--k 5 10] [--hanya kunci ...]
          python3 bandingkan_reranker2.py --ringkas   (gabungkan checkpoint)
"""

from __future__ import annotations

import argparse
import gc
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

CSV = HERE / "evaluasi-set-507.csv"
INDEX = HERE / "rag_cache" / "corpus31" / "bm25_index.pkl"
OUTDIR = HERE / "Result" / "2026-09-13_reranker2"
CKPT = OUTDIR / "checkpoint"

# (kunci, nama HF, bahasa, data latih, perlu trust_remote_code)
MODEL = [
    ("ms-marco-L6",  "cross-encoder/ms-marco-MiniLM-L-6-v2",
     "BERT-en 30k", "MS MARCO", False),
    ("mmarco-L6",    "unicamp-dl/mMiniLM-L6-v2-mmarco-v2",
     "XLM-R 250k", "mMARCO", False),
    ("bge-large-en", "BAAI/bge-reranker-large",
     "XLM-R 250k", "ft en/zh", False),
    ("bge-v2-m3",    "BAAI/bge-reranker-v2-m3",
     "XLM-R 250k", "beragam", False),
    ("ms-marco-L12", "cross-encoder/ms-marco-MiniLM-L-12-v2",
     "BERT-en 30k", "MS MARCO", False),
    ("mmarco-L12",   "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
     "XLM-R 250k", "mMARCO", False),
    ("gte-base",     "Alibaba-NLP/gte-multilingual-reranker-base",
     "mGTE 250k", "beragam", True),
    ("bge-base",     "BAAI/bge-reranker-base",
     "XLM-R 250k", "beragam", False),
    ("gte-mbert-en", "Alibaba-NLP/gte-reranker-modernbert-base",
     "ModernBERT-en 50k", "beragam", True),
    ("jina-v2",      "jinaai/jina-reranker-v2-base-multilingual",
     "XLM-R 250k", "beragam", True),
    ("mxbai-v2",     "mixedbread-ai/mxbai-rerank-base-v2",
     "Qwen2 152k", "beragam", True),
]
INFO = {k: {"kunci": k, "model": n, "bahasa": b, "data_latih": d}
        for k, n, b, d, _ in MODEL}


def tag(d) -> str:
    md = d.metadata or {}
    return f"{md.get('source', '')}:p{md.get('page')}"


def metrik_urutan(seq, r, k_list) -> dict:
    """Metrik satu urutan dokumen, untuk semua k dan kedua granularitas."""
    mode = str(getattr(r, "gold_mode", "all") or "all")
    out = {}
    for gran, pakai_hal in (("halaman", True), ("dokumen", False)):
        gold = {normalize_tag(g, include_page_suffix=pakai_hal)
                for g in parse_gold_set(str(r.gold_label))}
        for k in k_list:
            top = normalize_tag_list([tag(d) for d in seq[:k]],
                                     include_page_suffix=pakai_hal)
            m = rank_metrics(top, gold, k=k, gold_mode=mode)
            out[f"{gran}_k{k}_hit"] = m[f"hits@{k}"]
            out[f"{gran}_k{k}_rr"] = m["rr"]
            out[f"{gran}_k{k}_ndcg"] = m[f"ndcg@{k}"]
    return out


def muat_pertanyaan(n: int) -> pd.DataFrame:
    df = pd.read_csv(CSV, sep=";")
    df = df[df["answerable"] == "yes"].copy()
    if n and n < len(df):
        df = df.sample(n, random_state=42)
    return df


def ringkas(k_list) -> int:
    """Gabungkan semua checkpoint jadi satu tabel + uji Wilcoxon."""
    from scipy.stats import wilcoxon

    f_acuan = CKPT / "tanpa_rerank.csv"
    if not f_acuan.exists():
        raise SystemExit("[FATAL] checkpoint acuan belum ada")
    acuan = pd.read_csv(f_acuan).set_index("id")

    ring, punya = [], []
    for kunci in ["tanpa_rerank"] + [m[0] for m in MODEL]:
        f = CKPT / f"{kunci}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).set_index("id")
        # setiap model harus dinilai pada pertanyaan yang sama persis
        if not d.index.equals(acuan.index):
            print(f"[LEWAT] {kunci}: himpunan pertanyaan berbeda dari acuan")
            continue
        punya.append(kunci)
        for gran in ("halaman", "dokumen"):
            for k in k_list:
                pre = f"{gran}_k{k}"
                b = {"granularitas": gran, "k": k, "kondisi": kunci,
                     "n": len(d),
                     "juta_param": INFO.get(kunci, {}).get("juta_param", np.nan),
                     "bahasa": INFO.get(kunci, {}).get("bahasa", "-"),
                     "data_latih": INFO.get(kunci, {}).get("data_latih", "-"),
                     "HIT@K": d[f"{pre}_hit"].mean(),
                     "MRR": d[f"{pre}_rr"].mean(),
                     "NDCG@K": d[f"{pre}_ndcg"].mean()}
                if kunci != "tanpa_rerank":
                    for met, kol in (("rr", "MRR"), ("ndcg", "NDCG@K")):
                        a, c = acuan[f"{pre}_{met}"], d[f"{pre}_{met}"]
                        b[f"d_{kol}"] = c.mean() - a.mean()
                        b[f"p_{kol}"] = (wilcoxon(c, a, zero_method="wilcox")[1]
                                         if (c.values - a.values).any() else np.nan)
                ring.append(b)

    if not ring:
        raise SystemExit("[FATAL] tidak ada checkpoint untuk diringkas")
    r = pd.DataFrame(ring)
    # juta_param dari file info kalau ada
    f_info = OUTDIR / "info_model.csv"
    if f_info.exists():
        p = pd.read_csv(f_info).set_index("kunci")["juta_param"]
        r["juta_param"] = r["kondisi"].map(p)
    r.to_csv(OUTDIR / "ringkasan.csv", index=False)

    for gran in ("halaman", "dokumen"):
        for k in k_list:
            s = r[(r.granularitas == gran) & (r.k == k)].copy()
            s = s.sort_values("MRR", ascending=False)
            print(f"\n{'='*100}\nGRANULARITAS {gran.upper()}, K={k}, "
                  f"n={int(s['n'].iloc[0])}\n")
            print(f"  {'kondisi':14s}{'param':>7s} {'bahasa':18s}{'latih':10s}"
                  f"{'HIT@K':>8s}{'MRR':>8s}{'nDCG':>8s}{'dMRR':>9s}{'p(MRR)':>11s}")
            for _, x in s.iterrows():
                pr = "-" if pd.isna(x.juta_param) else f"{x.juta_param:.0f}M"
                if x.kondisi == "tanpa_rerank":
                    print(f"  {x.kondisi:14s}{pr:>7s} {'-':18s}{'-':10s}"
                          f"{x['HIT@K']:8.4f}{x['MRR']:8.4f}{x['NDCG@K']:8.4f}"
                          f"{'(acuan)':>20s}")
                else:
                    tanda = "" if pd.isna(x.p_MRR) else (" *" if x.p_MRR < 0.05 else "")
                    print(f"  {x.kondisi:14s}{pr:>7s} {x.bahasa:18s}{x.data_latih:10s}"
                          f"{x['HIT@K']:8.4f}{x['MRR']:8.4f}{x['NDCG@K']:8.4f}"
                          f"{x.d_MRR:+9.4f}{x.p_MRR:11.2e}{tanda}")
            print("\n  * = beda signifikan terhadap 'tanpa rerank' (Wilcoxon, p<0,05)")
    print(f"\nModel terangkum: {', '.join(punya)}")
    print(f"Keluaran: {OUTDIR/'ringkasan.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--n", type=int, default=0,
                    help="jumlah pertanyaan; 0 = semua yang berjawab")
    ap.add_argument("--fetch", type=int, default=48)
    ap.add_argument("--max-chars", type=int, default=1024)
    ap.add_argument("--hanya", nargs="+", default=None)
    ap.add_argument("--ulang", action="store_true",
                    help="hitung ulang walau checkpoint sudah ada")
    ap.add_argument("--ringkas", action="store_true",
                    help="hanya gabungkan checkpoint yang sudah ada")
    args = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)

    if args.ringkas:
        return ringkas(args.k)

    index = BM25Index.load(INDEX)
    if index is None:
        raise SystemExit(f"[FATAL] gagal memuat index {INDEX}")
    df = muat_pertanyaan(args.n)
    print(f"pertanyaan : {len(df)}")
    print(f"kandidat   : top-{args.fetch} BM25 -> dinilai ulang -> top-{args.k}")
    print(f"index      : {len(index.docs)} chunk\n", flush=True)

    # --- kandidat BM25 dihitung sekali; deterministik, dipakai semua model ---
    t = time.perf_counter()
    kandidat, urut_id = [], []
    for r in df.itertuples(index=False):
        docs = [d for d, _ in index.search(str(r.question), topn=args.fetch)]
        if not docs:
            continue
        kandidat.append((r, docs))
        urut_id.append(r.id)
    print(f"kandidat BM25 siap untuk {len(kandidat)} pertanyaan "
          f"({time.perf_counter()-t:.1f}s)\n", flush=True)

    # --- acuan tanpa rerank ---
    f = CKPT / "tanpa_rerank.csv"
    if args.ulang or not f.exists():
        baris = [{"id": r.id, "question_type": r.question_type,
                  **metrik_urutan(docs, r, args.k)} for r, docs in kandidat]
        pd.DataFrame(baris).to_csv(f, index=False)
        print("acuan tanpa_rerank disimpan\n", flush=True)

    from sentence_transformers import CrossEncoder

    dipakai = [m for m in MODEL if not args.hanya or m[0] in args.hanya]
    for kunci, nama, bahasa, latih, trc in dipakai:
        f = CKPT / f"{kunci}.csv"
        if f.exists() and not args.ulang:
            print(f"{kunci:14s} sudah ada, dilewati", flush=True)
            continue

        t = time.perf_counter()
        try:
            m = CrossEncoder(nama, **({"trust_remote_code": True} if trc else {}))
        except Exception as e:
            print(f"{kunci:14s} [LEWAT] {type(e).__name__}: {str(e)[:60]}",
                  flush=True)
            continue
        juta = sum(x.numel() for x in m.model.parameters()) / 1e6
        INFO[kunci]["juta_param"] = round(juta, 1)
        print(f"{kunci:14s} {juta:6.0f}M  {bahasa:18s} dimuat "
              f"{time.perf_counter()-t:.1f}s", flush=True)

        t0, baris = time.perf_counter(), []
        for i, (r, docs) in enumerate(kandidat, start=1):
            teks = [(d.page_content or "")[:args.max_chars] for d in docs]
            skor = m.predict([(str(r.question), x) for x in teks],
                             show_progress_bar=False)
            seq = [docs[j] for j in np.argsort(-np.asarray(skor))]
            baris.append({"id": r.id, "question_type": r.question_type,
                          **metrik_urutan(seq, r, args.k)})
            if i % 100 == 0:
                lalu = time.perf_counter() - t0
                print(f"    {i}/{len(kandidat)}  {lalu/i:.2f}s/pertanyaan, "
                      f"sisa {lalu/i*(len(kandidat)-i)/60:.1f} menit", flush=True)
        pd.DataFrame(baris).to_csv(f, index=False)
        dt = time.perf_counter() - t0
        print(f"{kunci:14s} selesai {dt/60:.1f} menit "
              f"({dt/len(kandidat)*1000:.0f} ms/pertanyaan)\n", flush=True)

        # bebaskan memori sebelum model berikutnya
        del m
        gc.collect()

    pd.DataFrame([v for v in INFO.values() if "juta_param" in v]).to_csv(
        OUTDIR / "info_model.csv", index=False)
    return ringkas(args.k)


if __name__ == "__main__":
    sys.exit(main())
