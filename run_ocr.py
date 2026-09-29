"""run_ocr.py - menambahkan text layer ke 5 PDF Peraturan Rektor hasil scan.

Kelima PDF ini tidak punya text layer sama sekali, sehingga PyPDFLoader
mengekstrak 0 karakter dan dokumennya tidak pernah terindeks.

Syarat yang dijaga skrip ini:
  1. Nama file keluaran SAMA PERSIS dengan nama masukan, karena gold_label
     di evaluasi-set-250.csv memakai nama file tersebut.
  2. Jumlah dan urutan halaman TIDAK berubah, karena gold_label memakai
     indeks halaman 0-based (":pN").

Jalankan: python3 run_ocr.py [--src DIR] [--out DIR]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

DOKUMEN = [
    "4-Penggunaan-AI-dalam-Kegiatan-Belajar-Mengajar.pdf",
    "5-Penerbitan-Ijazah-Transkrip-Akademik-Sertifikat-Profesi-dan-Surat-Keterangan-Pendamping-Ijazah.pdf",
    "7-Integritas-Akademik-dalam-Menghasilkan-Karya-Ilmiah.pdf",
    "8-Fast-Track.pdf",
    "9-Perubahan-atas-Peraturan-Rektor-Nomor-8-Tahun-2024-tentang-Penyelenggaraan-Pendidikan-Universitas-Riau.pdf",
]

BAHASA = "ind"


def jumlah_halaman(path: Path) -> int:
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def panjang_teks(path: Path) -> tuple[int, int]:
    """Kembalikan (total karakter, jumlah halaman yang ada teksnya)."""
    from pypdf import PdfReader
    total, berisi = 0, 0
    for hal in PdfReader(str(path)).pages:
        t = hal.extract_text() or ""
        total += len(t)
        if t.strip():
            berisi += 1
    return total, berisi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "Downloads"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "corpus"))
    ap.add_argument("--tessdata", default="")
    args = ap.parse_args()

    if not shutil.which("ocrmypdf"):
        print("[FATAL] ocrmypdf belum terpasang. Jalankan: brew install ocrmypdf")
        return 1

    env = dict(os.environ)
    if args.tessdata:
        env["TESSDATA_PREFIX"] = args.tessdata

    tersedia = subprocess.run(["tesseract", "--list-langs"], capture_output=True,
                              text=True, env=env).stdout
    if BAHASA not in tersedia.split():
        print(f"[FATAL] model bahasa '{BAHASA}' tidak ditemukan tesseract.")
        print(f"        bahasa tersedia: {' '.join(tersedia.split()[1:]) or '(kosong)'}")
        return 1

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    gagal = []
    for nama in DOKUMEN:
        masuk, keluar = src / nama, out / nama
        if not masuk.exists():
            print(f"[LEWAT] tidak ditemukan: {masuk}")
            gagal.append(nama)
            continue

        hal_asli = jumlah_halaman(masuk)
        char_asli, _ = panjang_teks(masuk)
        print(f"\n{nama[:70]}")
        print(f"  sebelum : {hal_asli} halaman, {char_asli} karakter")

        # --force-ocr menjamin OCR dijalankan meski ada sisa text layer kosong,
        # dan hasilnya deterministik untuk PDF berbasis gambar.
        cmd = ["ocrmypdf", "-l", BAHASA, "--force-ocr", "--output-type", "pdf",
               "--quiet", str(masuk), str(keluar)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(f"  [GAGAL] ocrmypdf keluar dengan kode {r.returncode}")
            print("  " + (r.stderr or "").strip()[:400])
            gagal.append(nama)
            continue

        hal_baru = jumlah_halaman(keluar)
        char_baru, hal_berisi = panjang_teks(keluar)
        print(f"  sesudah : {hal_baru} halaman, {char_baru} karakter "
              f"({hal_berisi}/{hal_baru} halaman ada teks)")

        if hal_baru != hal_asli:
            print(f"  [GAGAL] jumlah halaman berubah {hal_asli} -> {hal_baru}, "
                  f"gold_label :pN jadi tidak valid")
            gagal.append(nama)
        elif char_baru < 500:
            print("  [GAGAL] teks hasil OCR terlalu sedikit")
            gagal.append(nama)
        else:
            print("  OK")

    print("\n" + "=" * 60)
    if gagal:
        print(f"GAGAL pada {len(gagal)} dokumen: {', '.join(n[:40] for n in gagal)}")
        return 1
    print(f"Selesai: {len(DOKUMEN)} dokumen ber-text-layer di {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
