"""check_gold_conflicts.py - mendeteksi gold_label yang jadi tidak lengkap.

Menambah dokumen ke korpus bisa menciptakan false negative: sistem menemukan
jawaban yang benar, tapi di dokumen yang tidak terdaftar sebagai gold, lalu
dihitung salah oleh rank_metrics().

Skrip ini membandingkan, untuk setiap pertanyaan, seberapa banyak kata isi
reference_answer muncul di:
  - halaman gold yang terdaftar (skor acuan)
  - setiap halaman dokumen TAMBAHAN

Bila ada halaman dokumen tambahan yang skornya mendekati atau melampaui skor
gold, pertanyaan itu perlu ditinjau: gold-nya kemungkinan harus diperluas,
atau dokumen tambahan itu tidak layak dipakai sebagai distractor.

Jalankan: python3 check_gold_conflicts.py --tambahan DIR
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

STOP = {
    "adalah", "dalam", "untuk", "yang", "dengan", "pada", "dari", "atau", "dan",
    "oleh", "sebagai", "tidak", "dapat", "akan", "sudah", "telah", "harus",
    "paling", "setiap", "serta", "juga", "bagi", "para", "itu", "ini", "ada",
    "lebih", "sesuai", "ketentuan", "peraturan", "melalui", "secara", "yaitu",
    "antara", "lain", "terhadap", "maupun", "tersebut", "dimaksud", "berupa",
}

# Ambang: halaman dokumen tambahan dianggap bermasalah bila cakupannya
# mencapai minimal rasio ini terhadap cakupan halaman gold.
RASIO_KONFLIK = 0.90
MIN_ABSOLUT = 0.50


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def kata_isi(jawaban: str) -> list[str]:
    return [w for w in norm(jawaban).split()
            if w not in STOP and (w.isdigit() or len(w) >= 5)]


def muat(folder: Path) -> dict[str, list[str]]:
    from pypdf import PdfReader
    hasil = {}
    for p in sorted(folder.glob("*.pdf")):
        hasil[p.name] = [norm(x.extract_text() or "")
                         for x in PdfReader(str(p)).pages]
    return hasil


def cakupan(kata: list[str], teks: str) -> float:
    if not kata:
        return 0.0
    return sum(1 for w in kata if w in teks) / len(kata)


def frasa_terpanjang(jawaban: str, teks: str) -> int:
    """Panjang frasa BERURUTAN terpanjang dari jawaban yang ada di teks.

    Kantong kata terlalu longgar: jawaban pendek seperti "4-8 SKS" hanya
    menyisakan token "4" dan "8", sehingga halaman apa pun yang memuat kedua
    angka itu terlihat cocok sempurna. Kecocokan frasa berurutan jauh lebih
    kuat sebagai bukti bahwa jawaban benar-benar ada di halaman tersebut.
    """
    kata = norm(jawaban).split()
    if not kata:
        return 0
    terbaik = 0
    for i in range(len(kata)):
        # perpanjang selama frasa masih ditemukan
        j = i + terbaik
        while j < len(kata):
            frasa = " ".join(kata[i:j + 1])
            if frasa in teks:
                terbaik = max(terbaik, j - i + 1)
                j += 1
            else:
                break
    return terbaik


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tambahan", required=True,
                    help="folder berisi dokumen tambahan yang akan diuji")
    ap.add_argument("--csv", default=str(CSV))
    args = ap.parse_args()

    lama = muat(CORPUS)
    baru = muat(Path(args.tambahan))
    print(f"korpus gold  : {len(lama)} dokumen")
    print(f"dokumen baru : {len(baru)} dokumen "
          f"({sum(len(v) for v in baru.values())} halaman)\n")

    baris = list(csv.DictReader(open(args.csv, encoding="utf-8"), delimiter=";"))
    konflik, aman, tanpa_gold = [], 0, 0

    for r in baris:
        if not r["gold_label"]:
            tanpa_gold += 1
            continue
        kata = kata_isi(r["reference_answer"])
        if not kata:
            continue

        jwb = r["reference_answer"]

        # acuan: frasa berurutan terpanjang yang cocok di halaman gold
        acuan = 0
        for bagian in r["gold_label"].split("|"):
            if ":p" in bagian:
                nama, hal = bagian.rsplit(":p", 1)
                if nama in lama and int(hal) < len(lama[nama]):
                    acuan = max(acuan, frasa_terpanjang(jwb, lama[nama][int(hal)]))
            elif bagian in lama:
                acuan = max(acuan, max(frasa_terpanjang(jwb, t) for t in lama[bagian]))

        # frasa terpanjang yang cocok di dokumen tambahan
        terbaik, dimana = 0, ""
        for nama, halaman in baru.items():
            for i, t in enumerate(halaman):
                n = frasa_terpanjang(jwb, t)
                if n > terbaik:
                    terbaik, dimana = n, f"{nama}:p{i}"

        # Konflik bila dokumen tambahan memuat frasa panjang (>= 6 kata
        # berurutan) yang setara atau lebih panjang daripada di halaman gold.
        if terbaik >= 6 and terbaik >= acuan:
            konflik.append((terbaik, acuan, r["id"], r["question"][:52], dimana))
        else:
            aman += 1

    print("=" * 78)
    print(f"aman        : {aman}")
    print(f"perlu tinjau: {len(konflik)}")
    print(f"unanswerable: {tanpa_gold}")

    if konflik:
        per_dok: dict[str, int] = {}
        for _, _, _, _, dimana in konflik:
            per_dok[dimana.split(":p")[0]] = per_dok.get(dimana.split(":p")[0], 0) + 1
        print("\nDokumen tambahan penyebab konflik:")
        for nama, n in sorted(per_dok.items(), key=lambda x: -x[1]):
            print(f"  {n:4d}  {nama[:64]}")

        print("\nPertanyaan yang perlu ditinjau (cakupan tertinggi dulu):")
        print(f"  {chr(102)+chr(114)+chr(97)+chr(115)+chr(97):>5s} {chr(103)+chr(111)+chr(108)+chr(100):>5s}  id     pertanyaan")
        for c, a, qid, q, dimana in sorted(konflik, reverse=True):
            print(f"  {c:5d} {a:5d}  {qid}  {q}")
            print(f"                       -> {dimana[:70]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
