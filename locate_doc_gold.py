"""locate_doc_gold.py - menaikkan gold level dokumen menjadi level halaman.

40 pertanyaan lama masih bergold level dokumen (tanpa ":pN"). Akibatnya:

  - evaluasi level halaman tidak bisa menyertakannya sama sekali, sehingga
    sweep harus dipecah menjadi dua run dengan populasi berbeda;
  - pada figure cakupan, dokumennya terbaca "0 halaman diuji" seolah-olah
    distractor, padahal justru dipakai.

Skrip ini mencari halaman yang benar-benar memuat jawabannya dengan
mencocokkan frasa BERURUTAN terpanjang dari reference_answer ke tiap halaman
dokumen gold, lalu melaporkan keyakinannya. Keluarannya adalah usulan yang
harus ditinjau, bukan perubahan yang langsung diterapkan.

Jalankan: python3 locate_doc_gold.py [--min-frasa 4] [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluasi-set-507.csv"
CORPUS = HERE / "corpus"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def frasa_terpanjang(jawaban: str, teks: str) -> int:
    kata = norm(jawaban).split()
    if not kata:
        return 0
    terbaik = 0
    for i in range(len(kata)):
        j = i + terbaik
        while j < len(kata):
            if " ".join(kata[i:j + 1]) in teks:
                terbaik = j - i + 1
                j += 1
            else:
                break
    return terbaik


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-frasa", type=int, default=4,
                    help="panjang frasa minimum agar sebuah halaman dipercaya")
    ap.add_argument("--out", default=str(HERE / "Result" / "eda" / "usulan_gold_halaman.csv"))
    args = ap.parse_args()

    from pypdf import PdfReader

    rows = list(csv.DictReader(open(CSV, encoding="utf-8"), delimiter=";"))
    perlu = [r for r in rows
             if r["gold_label"] and any(":p" not in p for p in r["gold_label"].split("|"))]
    print(f"pertanyaan dengan gold level dokumen: {len(perlu)}\n")

    halaman: dict[str, list[str]] = {}
    for p in sorted(CORPUS.glob("*.pdf")):
        halaman[p.name] = [norm(x.extract_text() or "") for x in PdfReader(str(p)).pages]

    usulan, ragu = [], []
    for r in perlu:
        bagian_baru = []
        catatan = []
        for bagian in r["gold_label"].split("|"):
            if ":p" in bagian:
                bagian_baru.append(bagian)
                continue
            nm = bagian
            if nm not in halaman:
                bagian_baru.append(bagian)
                catatan.append(f"{nm}: tidak ada di corpus")
                continue
            skor = [(frasa_terpanjang(r["reference_answer"], t), i)
                    for i, t in enumerate(halaman[nm])]
            skor.sort(reverse=True)
            n1, h1 = skor[0]
            n2 = skor[1][0] if len(skor) > 1 else 0
            if n1 >= args.min_frasa:
                bagian_baru.append(f"{nm}:p{h1}")
                catatan.append(f"{nm}:p{h1} frasa={n1} (kedua={n2})")
            else:
                bagian_baru.append(bagian)      # biarkan level dokumen
                catatan.append(f"{nm}: TIDAK YAKIN, frasa terbaik={n1} di p{h1}")
                ragu.append((r["id"], nm, n1, h1, r["question"][:50]))
        usulan.append({
            "id": r["id"], "question": r["question"],
            "gold_lama": r["gold_label"], "gold_usul": "|".join(bagian_baru),
            "berubah": "ya" if "|".join(bagian_baru) != r["gold_label"] else "tidak",
            "catatan": "; ".join(catatan),
        })

    berubah = [u for u in usulan if u["berubah"] == "ya"]
    print(f"berhasil dinaikkan ke level halaman : {len(berubah)}")
    print(f"masih ragu, dibiarkan level dokumen : {len(ragu)}\n")

    if ragu:
        print("Perlu ditinjau manual:")
        for qid, nm, n, h, q in ragu:
            print(f"  {qid}  frasa terbaik {n} kata di p{h}  {nm[:34]}")
            print(f"        {q}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(usulan[0].keys()))
        w.writeheader()
        w.writerows(usulan)
    print(f"\nUsulan lengkap: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
