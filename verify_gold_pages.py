"""verify_gold_pages.py - memeriksa gold_label level halaman terhadap teks OCR.

Untuk setiap baris evaluasi-set-250.csv yang gold-nya menunjuk halaman
tertentu di corpus/, skrip mengecek apakah kata-kata isi dari
reference_answer benar-benar muncul di halaman tersebut.

Ini menguji dua hal sekaligus:
  - kualitas OCR (teksnya terbaca atau tidak)
  - ketepatan gold_label (halamannya benar atau tidak)

Jalankan: python3 verify_gold_pages.py [--csv FILE] [--semua-entri]

--semua-entri memeriksa SETIAP entri gold pada baris ber-gold ganda, bukan
hanya entri pertama. Untuk gold_mode "all" itulah yang benar, karena kedua
halaman memang harus sama-sama memuat bagian jawabannya.
"""

from __future__ import annotations

import csv
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluasi-set-507.csv"
CORPUS = HERE / "corpus"

# Kata umum yang tidak membedakan apa-apa.
STOP = {
    "adalah", "dalam", "untuk", "yang", "dengan", "pada", "dari", "atau", "dan",
    "oleh", "sebagai", "tidak", "dapat", "akan", "sudah", "telah", "harus",
    "paling", "setiap", "serta", "juga", "bagi", "para", "itu", "ini", "ada",
    "lebih", "sesuai", "ketentuan", "peraturan", "melalui", "secara", "yaitu",
    "antara", "lain", "terhadap", "maupun", "tersebut", "dimaksud", "berupa",
}

AMBANG = 0.45   # minimal 45% kata isi jawaban harus muncul di halaman gold


def normalisasi(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def kata_isi(jawaban: str) -> list[str]:
    kata = []
    for w in normalisasi(jawaban).split():
        if w in STOP:
            continue
        if w.isdigit() or len(w) >= 5:
            kata.append(w)
    return kata


def main() -> int:
    import argparse
    from pypdf import PdfReader

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--semua-entri", action="store_true")
    args = ap.parse_args()
    berkas = Path(args.csv)

    if not CORPUS.exists():
        print(f"[FATAL] {CORPUS} belum ada. Jalankan run_ocr.py dulu.")
        return 1

    halaman: dict[str, list[str]] = {}
    for pdf in sorted(CORPUS.glob("*.pdf")):
        halaman[pdf.name] = [normalisasi(p.extract_text() or "")
                             for p in PdfReader(str(pdf)).pages]
        print(f"dimuat: {pdf.name[:60]:60s} {len(halaman[pdf.name])} halaman")

    baris = list(csv.DictReader(open(berkas, encoding="utf-8"), delimiter=";"))
    diperiksa, lolos, lemah, error = 0, 0, [], []

    for r in baris:
        if not r["gold_label"]:
            continue
        for bagian in r["gold_label"].split("|"):
            if ":p" not in bagian:
                continue
            nama, hal = bagian.rsplit(":p", 1)
            if nama not in halaman:
                continue
            idx = int(hal)
            if idx >= len(halaman[nama]):
                error.append(f"{r['id']}: halaman {idx} di luar rentang {nama[:40]}")
                continue

            kata = kata_isi(r["reference_answer"])
            if not kata:
                continue
            teks = halaman[nama][idx]
            cocok = sum(1 for w in kata if w in teks)
            rasio = cocok / len(kata)
            diperiksa += 1
            if rasio >= AMBANG:
                lolos += 1
            else:
                lemah.append((rasio, r["id"], bagian, r["question"][:55]))
            if not args.semua_entri:
                break   # cukup entri pertama

    print(f"\n{'=' * 66}")
    print(f"gold level halaman diperiksa : {diperiksa}")
    print(f"kata jawaban ditemukan >= {int(AMBANG*100)}% : {lolos} "
          f"({lolos/diperiksa*100:.1f}%)" if diperiksa else "")
    print(f"di bawah ambang              : {len(lemah)}")
    if error:
        print(f"halaman di luar rentang      : {len(error)}")
        for e in error[:5]:
            print("   ", e)

    if lemah:
        print(f"\nBaris yang perlu ditinjau manual (rasio terendah dulu):")
        for rasio, qid, gold, q in sorted(lemah)[:25]:
            print(f"  {rasio:5.2f}  {qid}  {gold.split(':p')[-1]:>3s}  {q}")

    print(f"\n{'=' * 66}")
    print("Spot-check fakta kunci yang dibaca manual dari PDF asli:")
    cek = [
        ("4-Penggunaan-AI-dalam-Kegiatan-Belajar-Mengajar.pdf", 6, "25", "batas 25% deteksi AI"),
        ("8-Fast-Track.pdf", 13, "500", "TOEFL minimal 500"),
        ("8-Fast-Track.pdf", 12, "144", "144 SKS program sarjana"),
        ("9-Perubahan-atas-Peraturan-Rektor-Nomor-8-Tahun-2024-tentang-"
         "Penyelenggaraan-Pendidikan-Universitas-Riau.pdf", 12, "108", "beban studi D3 108-120 SKS"),
        ("7-Integritas-Akademik-dalam-Menghasilkan-Karya-Ilmiah.pdf", 5, "fabrikasi", "jenis pelanggaran"),
        ("5-Penerbitan-Ijazah-Transkrip-Akademik-Sertifikat-Profesi-dan-"
         "Surat-Keterangan-Pendamping-Ijazah.pdf", 3, "450", "TOEFL 450 bagi S1"),
    ]
    gagal_spot = 0
    for nama, idx, istilah, label in cek:
        if nama not in halaman or idx >= len(halaman[nama]):
            print(f"  ??  {label}: dokumen/halaman tidak ada")
            gagal_spot += 1
            continue
        ada = istilah.lower() in halaman[nama][idx]
        print(f"  {'OK ' if ada else 'TIDAK'}  {label:32s} (cari {istilah!r} di :p{idx})")
        if not ada:
            gagal_spot += 1

    return 1 if (error or gagal_spot) else 0


if __name__ == "__main__":
    sys.exit(main())
