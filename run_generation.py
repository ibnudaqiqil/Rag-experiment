"""run_generation.py - eksperimen tahap generasi: kualitas jawaban dan abstensi.

Sweep retrieval hanya mengukur apakah dokumen yang benar terambil. Skrip ini
mengukur dua hal lain yang tidak bisa dijawab sweep tersebut:

1. Kualitas jawaban pada 342 pertanyaan berjawab: Exact Match, token-F1, dan
   kemiripan kosinus terhadap jawaban acuan.

2. Abstensi pada 8 pertanyaan unanswerable: apakah sistem menolak menjawab
   ketika jawabannya memang tidak ada di korpus. Dilaporkan juga abstensi
   palsu, yaitu sistem menolak menjawab padahal jawabannya ada - kesalahan
   yang sama merugikannya tapi hampir tidak pernah dilaporkan.

Setiap jawaban yang dihasilkan disimpan, baik di llm_cache/responses.jsonl
maupun di CSV keluaran, sehingga dapat diperiksa manual.

Jalankan: python3 run_generation.py [--k 10] [--rerank] [--limit N]
"""

from __future__ import annotations

import argparse
import os
import re
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

import llm_cache  # noqa: E402
from bm25_index import BM25Index  # noqa: E402
from evaluator import generate_answer  # noqa: E402
from utils import compute_answer_match, cosine_similarity_answer  # noqa: E402

CSV = HERE / "evaluasi-set-507.csv"
INDEX = HERE / "rag_cache" / "corpus31" / "bm25_index.pkl"
OUTDIR = HERE / "Result" / "2026-09-13_generasi"

# Frasa yang menandakan sistem menolak menjawab. Dicocokkan pada teks jawaban
# yang sudah dinormalisasi. Prompt generate_answer memang meminta model
# menyatakan bila konteksnya tidak memadai.
POLA_ABSTAIN = re.compile(
    r"tidak (tersedia|ada|ditemukan|terdapat|memuat|menyebut|cukup|memadai|"
    r"disebutkan|dijelaskan|diatur)"
    r"|belum (tersedia|diatur|disebutkan)"
    r"|insufficient|not (found|available|mentioned|specified|provided)"
    r"|no (information|mention|relevant)"
    r"|cannot (answer|determine|find)"
    r"|informasi.{0,30}tidak", re.I)


def klasifikasi_abstain(teks: str) -> str:
    """Bedakan penolakan penuh dari sekadar catatan kehati-hatian.

    Pencocokan pola saja tidak cukup. Model sering menjawab lengkap lalu
    menambah kalimat seperti "informasi lebih spesifik tidak tersedia".
    Itu hedging, bukan penolakan; menghitungnya sebagai abstensi membuat
    sistem yang berhati-hati tampak seolah selalu menolak menjawab.

    Kembalikan salah satu: 'penuh', 'sebagian', atau 'menjawab'.
    """
    t = (teks or "").strip()
    if not t:
        return "penuh"
    kalimat = [k.strip() for k in re.split(r"(?<=[.!?])\s+|\n+", t) if k.strip()]
    if not kalimat:
        return "menjawab"
    cocok = [bool(POLA_ABSTAIN.search(k)) for k in kalimat]
    if not any(cocok):
        return "menjawab"
    # penolakan penuh: muncul di kalimat pertama, atau jawabannya memang pendek
    # dan tidak memuat isi substantif selain penolakan
    if cocok[0] or len(t) < 160 or all(cocok):
        return "penuh"
    return "sebagian"


def abstain(teks: str) -> bool:
    """Kompatibilitas: hanya penolakan penuh yang dihitung abstensi."""
    return klasifikasi_abstain(teks) == "penuh"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank", action="store_true", help="pakai cross-encoder reranker")
    ap.add_argument("--provider", default="openai", choices=["openai", "gemini"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cosine-thr", type=float, default=0.75)
    ap.add_argument("--f1-thr", type=float, default=0.5)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    if args.provider == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise SystemExit("[FATAL] OPENAI_API_KEY kosong")
        llm = ChatOpenAI(model=args.model, temperature=0.2, timeout=90, max_retries=4)
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=args.model, temperature=0.2,
                                     timeout=90, max_retries=4)

    index = BM25Index.load(INDEX)
    if index is None:
        raise SystemExit(f"[FATAL] gagal memuat index {INDEX}")

    rerank_fn = None
    if args.rerank:
        from reranker import rerank_crossencoder
        rerank_fn = rerank_crossencoder

    df = pd.read_csv(CSV, sep=";")
    if args.limit:
        df = df.head(args.limit)

    print(f"index    : {len(index.docs)} chunk")
    print(f"dataset  : {len(df)} pertanyaan "
          f"({int((df.answerable == 'no').sum())} unanswerable)")
    print(f"k        : {args.k}   reranker: {'ON' if rerank_fn else 'OFF'}")
    import hashlib
    sha = hashlib.sha256(CSV.read_bytes()).hexdigest()[:12]
    print(f"dataset  : sha {sha}")
    print(f"model    : {args.provider}/{args.model}\n", flush=True)

    baris = []
    t0 = time.perf_counter()
    for i, r in enumerate(df.itertuples(index=False), start=1):
        q = str(r.question)
        pasangan = index.search(q, topn=max(args.k * 4, 40))
        docs = [d for d, _ in pasangan]
        if rerank_fn is not None:
            docs, _ = rerank_fn(q, docs, top_k=args.k)
        docs = docs[:args.k]

        jawab, _ = generate_answer(llm, q, docs)
        acuan = str(r.reference_answer)
        cos = cosine_similarity_answer(jawab, acuan)
        m = compute_answer_match(jawab, acuan, cos_sim=cos, use_em=True,
                                 cosine_thr=args.cosine_thr, f1_thr=args.f1_thr)

        baris.append({
            "id": r.id, "question": q, "answerable": r.answerable,
            "question_type": r.question_type, "difficulty": r.difficulty,
            "reference_answer": acuan, "generated_answer": jawab,
            "abstain": abstain(jawab),
            "abstain_kelas": klasifikasi_abstain(jawab),
            "answer_em": m["em"], "answer_f1": m["f1"],
            "answer_correct": m["correct"], "cosine": cos,
        })
        if i % 25 == 0 or i == len(df):
            print(f"  {i}/{len(df)}  {llm_cache.ringkas()}", flush=True)

    hasil = pd.DataFrame(baris)
    out = OUTDIR / f"generasi_k{args.k}_{'rerank' if rerank_fn else 'norerank'}.csv"
    hasil.to_csv(out, index=False)

    jwb = hasil[hasil["answerable"] == "yes"]
    tak = hasil[hasil["answerable"] == "no"]

    print(f"\n{'=' * 66}")
    print(f"KUALITAS JAWABAN ({len(jwb)} pertanyaan berjawab)")
    print(f"  Exact Match        : {jwb['answer_em'].mean():.4f}")
    print(f"  token-F1           : {jwb['answer_f1'].mean():.4f}")
    print(f"  kemiripan kosinus  : {jwb['cosine'].mean():.4f}")
    print(f"  benar (semantik)   : {jwb['answer_correct'].mean():.4f}")

    print(f"\nABSTENSI")
    if len(tak):
        benar_abstain = int(tak["abstain"].sum())
        print(f"  unanswerable       : {len(tak)} pertanyaan")
        print(f"  menolak menjawab   : {benar_abstain}/{len(tak)} "
              f"({benar_abstain/len(tak)*100:.0f}%)  <- makin tinggi makin baik")
        for _, r in tak.iterrows():
            tanda = "TOLAK " if r["abstain"] else "JAWAB "
            print(f"    {tanda} {r['id']}  {r['question'][:44]:44s} -> "
                  f"{str(r['generated_answer'])[:60]}")
    import collections as _c
    kelas = _c.Counter(jwb["abstain_kelas"])
    palsu = int(kelas.get("penuh", 0))
    print(f"\n  Pada {len(jwb)} pertanyaan BERJAWAB:")
    print(f"    menolak penuh    : {palsu} ({palsu/len(jwb)*100:.1f}%)  "
          f"<- abstensi palsu, makin rendah makin baik")
    print(f"    jawab + catatan  : {kelas.get('sebagian', 0)} "
          f"({kelas.get('sebagian', 0)/len(jwb)*100:.1f}%)  <- hedging, bukan penolakan")
    print(f"    menjawab langsung: {kelas.get('menjawab', 0)} "
          f"({kelas.get('menjawab', 0)/len(jwb)*100:.1f}%)")

    print(f"\nKualitas jawaban per tipe pertanyaan:")
    g = jwb.groupby("question_type").agg(
        n=("answer_f1", "size"), f1=("answer_f1", "mean"),
        kosinus=("cosine", "mean"), benar=("answer_correct", "mean"))
    for tipe, row in g.sort_values("f1", ascending=False).iterrows():
        print(f"  {tipe:14s} n={int(row['n']):4d}  F1={row['f1']:.4f}  "
              f"kosinus={row['kosinus']:.4f}  benar={row['benar']:.4f}")

    print(f"\nwaktu total: {(time.perf_counter()-t0)/60:.1f} menit")
    print(llm_cache.ringkas())
    print(f"Jawaban lengkap: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
