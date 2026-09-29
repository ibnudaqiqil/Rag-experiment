"""run_sweep.py - sweep 5 konfigurasi retrieval x K=5,10 untuk Tabel 1 paper.

Konfigurasi (sesuai Tabel 1):
  BM25, BM25+Multi-Query, BM25+HyDE, BM25+Multi-Query+Rerank, BM25+HyDE+Rerank

Dijalankan dua kali dengan granularitas gold berbeda:

  dokumen : seluruh 350 pertanyaan. gold_label dinormalisasi tanpa nomor
            halaman. Sebanding dengan hasil lama, tapi daya bedanya rendah
            karena chunk mana pun dari dokumen benar dihitung relevan.

  halaman : hanya 310 pertanyaan yang gold-nya menunjuk halaman. 40
            pertanyaan lama bergold level dokumen dikeluarkan karena tidak
            akan pernah cocok saat nomor halaman ikut dibandingkan.

Semua keluaran LLM disimpan di llm_cache/responses.jsonl dan dipakai ulang,
sehingga ekspansi Multi-Query/HyDE hanya dipanggil sekali per pertanyaan.

Jalankan: python3 run_sweep.py [--k 5 10] [--model NAMA] [--limit N]
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

from dotenv import load_dotenv  # noqa: E402
load_dotenv(HERE / ".env")

import pandas as pd  # noqa: E402

import hashlib  # noqa: E402

import llm_cache  # noqa: E402
from ablation import AblationSetup, run_ablation  # noqa: E402
from bm25_index import BM25Index  # noqa: E402

CSV = HERE / "evaluasi-set-507.csv"
INDEX = HERE / "rag_cache" / "corpus31" / "bm25_index.pkl"
OUTDIR = HERE / "Result" / "2026-09-13_sweep-507"

DATASET_SHA = ""
MODEL_NAME = ""

STRATEGI = {
    "BM25": ("BM25 (standar)", "OFF"),
    "BM25+MultiQuery": ("BM25 + Multi-Query", "OFF"),
    "BM25+HyDE": ("BM25 + HyDE", "OFF"),
    "BM25+MultiQuery+Rerank": ("BM25 + Multi-Query", "ON"),
    "BM25+HyDE+Rerank": ("BM25 + HyDE", "ON"),
    "BM25+Rerank": ("BM25 (standar)", "ON"),   # diagnostik: efek reranker murni
}


def buat_llm(provider: str, model: str):
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise SystemExit("[FATAL] OPENAI_API_KEY kosong")
        return ChatOpenAI(model=model, temperature=0.2, timeout=90, max_retries=4)
    from langchain_google_genai import ChatGoogleGenerativeAI
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        raise SystemExit("[FATAL] GOOGLE_API_KEY kosong")
    return ChatGoogleGenerativeAI(model=model, temperature=0.2,
                                  timeout=90, max_retries=4)


def jalankan(df, index, llm, k_list, page_level: bool, label: str,
             hanya=None) -> pd.DataFrame:
    hasil = []
    for nama, (strategi, rr) in STRATEGI.items():
        if hanya and nama not in hanya:
            continue
        for k in k_list:
            t0 = time.perf_counter()
            setup = AblationSetup(
                strategies=(strategi,), reranker_flags=(rr,), k_list=(k,),
                include_page_suffix=page_level, save_dir=None)
            detail, ringkas = run_ablation(df, index, setup=setup, llm=llm)
            baris = ringkas.iloc[0].to_dict()
            baris.update({"setup": nama, "k": k, "granularitas": label,
                          "dataset_sha": DATASET_SHA, "model": MODEL_NAME,
                          "reranker": os.getenv("RERANKER_MODEL",
                                                "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                          "n_samples": len(df),
                          "wall_s": round(time.perf_counter() - t0, 1)})
            hasil.append(baris)
            detail.to_csv(OUTDIR / f"detail_{label}_{nama}_k{k}.csv", index=False)
            print(f"  {nama:24s} k={k:<3d} "
                  f"HIT={baris.get(f'hits@{k}_avg', float('nan')):.4f} "
                  f"MRR={baris.get('mrr', float('nan')):.4f} "
                  f"MAP={baris.get('map', float('nan')):.4f} "
                  f"nDCG={baris.get(f'ndcg@{k}_avg', float('nan')):.4f} "
                  f"({baris['wall_s']}s)  {llm_cache.ringkas()}", flush=True)
    return pd.DataFrame(hasil)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah pertanyaan (uji coba)")
    ap.add_argument("--outdir", default=None,
                    help="folder keluaran lain, mis. saat menguji reranker berbeda")
    ap.add_argument("--reranker", default=None,
                    help="nama model reranker; default cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--setups", nargs="+", default=None,
                    help="jalankan sebagian setup saja, mis. saat kuota LLM habis")
    args = ap.parse_args()

    global OUTDIR
    if args.outdir:
        OUTDIR = Path(args.outdir)
    if args.reranker:
        # reranker.py membaca RERANKER_MODEL saat model pertama kali dimuat
        os.environ["RERANKER_MODEL"] = args.reranker
    OUTDIR.mkdir(parents=True, exist_ok=True)

    index = BM25Index.load(INDEX)
    if index is None:
        raise SystemExit(f"[FATAL] gagal memuat index {INDEX}")

    df = pd.read_csv(CSV, sep=";")
    if args.limit:
        df = df.head(args.limit)
    df_hal = df[df["gold_label"].astype(str).str.contains(":p", na=False)].copy()

    print(f"index      : {len(index.docs)} chunk")
    print(f"dataset    : {len(df)} pertanyaan ({len(df_hal)} bergold level halaman)")
    sha = hashlib.sha256(CSV.read_bytes()).hexdigest()[:12]
    print(f"dataset sha: {sha}")
    print(f"model LLM  : {args.provider}/{args.model}")
    print(f"reranker   : {os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')}")
    print(f"k          : {args.k}")
    print(f"output     : {OUTDIR}\n")

    global DATASET_SHA, MODEL_NAME
    DATASET_SHA, MODEL_NAME = sha, f"{args.provider}/{args.model}"
    llm = buat_llm(args.provider, args.model)

    print("=== granularitas DOKUMEN (seluruh pertanyaan) ===", flush=True)
    r_dok = jalankan(df, index, llm, args.k, page_level=False, label="dokumen", hanya=args.setups)

    print("\n=== granularitas HALAMAN (hanya gold level halaman) ===", flush=True)
    r_hal = jalankan(df_hal, index, llm, args.k, page_level=True, label="halaman", hanya=args.setups)

    semua = pd.concat([r_dok, r_hal], ignore_index=True)
    kolom = (["granularitas", "setup", "k", "n_samples"]
             + [c for c in semua.columns
                if c not in ("granularitas", "setup", "k", "n_samples")])
    semua = semua[kolom]
    out = OUTDIR / "sweep_summary.csv"
    semua.to_csv(out, index=False)

    # metrik yang terikat [0,1] tidak boleh terlampaui
    bat = [c for c in semua.columns if any(t in c.lower() for t in
           ("hits", "precision", "recall", "f1", "mrr", "map", "ndcg"))]
    buruk = [f"{c}={pd.to_numeric(semua[c], errors='coerce').max():.4f}"
             for c in bat if (pd.to_numeric(semua[c], errors="coerce") > 1.0001).any()]
    print(f"\nCek batas [0,1]: {'GAGAL -> ' + ', '.join(buruk) if buruk else 'OK'}")
    print(llm_cache.ringkas())
    print(f"Ringkasan: {out}")
    return 1 if buruk else 0


if __name__ == "__main__":
    sys.exit(main())
