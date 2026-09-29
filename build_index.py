"""build_index.py - membangun BM25Index dari seluruh PDF di corpus/.

Memakai parameter chunking yang sama dengan default app.py
(chunk_size=800, chunk_overlap=120) agar hasilnya sebanding.

Metadata 'source' diisi nama file dan 'page' diisi indeks halaman 0-based,
persis seperti yang diharapkan gold_label pada evaluasi-set-350.csv.

Jalankan: python3 build_index.py [--chunk-size N] [--overlap N]
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"
OUT = HERE / "rag_cache" / "corpus31" / "bm25_index.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader
    from bm25_index import BM25Index

    pdfs = sorted(CORPUS.glob("*.pdf"))
    if not pdfs:
        print(f"[FATAL] tidak ada PDF di {CORPUS}")
        return 1

    halaman = []
    for p in pdfs:
        sub = PyPDFLoader(str(p)).load()
        for d in sub:
            d.metadata["source"] = p.name      # gold_label memakai nama file
        kosong = sum(1 for d in sub if not (d.page_content or "").strip())
        halaman.extend(sub)
        tanda = "  <-- ADA HALAMAN KOSONG" if kosong else ""
        print(f"  {len(sub):4d} hal  {kosong:3d} kosong  {p.name[:58]}{tanda}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size, chunk_overlap=args.overlap, add_start_index=True)
    chunks = splitter.split_documents(halaman)

    print(f"\n{len(pdfs)} dokumen, {len(halaman)} halaman -> {len(chunks)} chunk "
          f"(chunk_size={args.chunk_size}, overlap={args.overlap})")

    per = collections.Counter((c.metadata or {}).get("source", "?") for c in chunks)
    print("\nChunk per dokumen:")
    for nama, n in per.most_common():
        print(f"  {n:5d}  {nama[:64]}")

    tanpa_page = sum(1 for c in chunks if (c.metadata or {}).get("page") is None)
    if tanpa_page:
        print(f"\n[PERINGATAN] {tanpa_page} chunk tanpa metadata 'page'; "
              f"evaluasi level halaman tidak akan cocok untuk chunk tersebut")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = BM25Index.build(chunks)
    index.save(out, fingerprint=f"corpus31|cs={args.chunk_size}|ov={args.overlap}")
    print(f"\nIndex disimpan: {out}  ({out.stat().st_size/1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
