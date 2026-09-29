"""contoh_keluaran_iaa.py - pratinjau bentuk laporan hitung_iaa.py.

DIPAKAI UNTUK APA
    Melihat seperti apa keluaran hitung_iaa.py sebelum lembar penilaian yang
    sebenarnya kembali dari ketiga penilai. Berguna untuk memastikan kolom,
    angka, dan berkas keluarannya sudah sesuai kebutuhan manuskrip.

TIDAK DIPAKAI UNTUK APA
    Angka yang dihasilkan skrip ini BUKAN hasil penilaian siapa pun. Penilaian
    ketiga "penilai" di sini dibangkitkan komputer dari label difficulty yang
    sudah ada, ditambah derau acak. Angka kappa dan alpha yang keluar TIDAK
    BOLEH masuk manuskrip dalam bentuk apa pun.

    Karena itu seluruh keluarannya ditaruh di folder terpisah bernama
    CONTOH_KELUARAN/, setiap berkasnya diberi awalan CONTOH_, dan barisnya
    diberi penanda simulasi. Folder itu juga tidak ikut di-commit.

Jalankan: python3 contoh_keluaran_iaa.py [--seed N] [--sepakat 0.72]
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASTER = HERE / "Result" / "anotasi" / "master_kesulitan.csv"
OUT = HERE / "Result" / "anotasi" / "CONTOH_KELUARAN"

SKALA = ["easy", "medium", "hard"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sepakat", type=float, default=0.72,
                    help="peluang seorang penilai sependapat dengan label dataset")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.master, encoding="utf-8")))
    rnd = random.Random(args.seed)

    def nilai_simulasi(acuan: str) -> str:
        """Sependapat dengan peluang --sepakat; kalau tidak, geser satu tingkat
        (kadang dua), karena penilai nyata lebih sering berselisih di batas
        yang berdekatan daripada melompat dari easy ke hard."""
        if rnd.random() < args.sepakat:
            return acuan
        i = SKALA.index(acuan) if acuan in SKALA else 1
        geser = rnd.choice([-1, 1] * 5 + [-2, 2])
        return SKALA[max(0, min(len(SKALA) - 1, i + geser))]

    for r in rows:
        acuan = (r.get("final") or "medium").strip().lower()
        for k in ("reviewer_1", "reviewer_2", "reviewer_3"):
            r[k] = nilai_simulasi(acuan)
        r["catatan_adjudikasi"] = "SIMULASI - bukan penilaian manusia"

    palsu = out / "CONTOH_master_terisi.csv"
    with open(palsu, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Penilaian simulasi ditulis ke {palsu.name}")
    print(f"  seed={args.seed}  peluang sependapat={args.sepakat}\n")
    print("=" * 70)
    print("CONTOH KELUARAN - ANGKA DI BAWAH INI HASIL SIMULASI, BUKAN DATA NYATA")
    print("=" * 70 + "\n")

    hasil = subprocess.run(
        [sys.executable, str(HERE / "hitung_iaa.py"),
         "--master", str(palsu), "--out", str(out)],
        capture_output=True, text=True)
    print(hasil.stdout, end="")
    if hasil.stderr:
        print(hasil.stderr, file=sys.stderr, end="")

    # Tandai juga di dalam berkas kalimatnya, supaya kalau berkasnya terbawa
    # sendirian pun masih jelas asalnya.
    kalimat = out / "kalimat_iaa.txt"
    if kalimat.exists():
        isi = kalimat.read_text(encoding="utf-8")
        kalimat.write_text(
            "[CONTOH - angka dari penilaian simulasi, JANGAN dipakai di manuskrip]\n\n"
            + isi, encoding="utf-8")

    print("\n" + "=" * 70)
    print("Sekali lagi: angka di atas dibangkitkan komputer dari label yang")
    print("sudah ada. Kappa yang sah baru muncul setelah ketiga penilai")
    print("mengembalikan lembarnya dan hitung_iaa.py dijalankan atas isinya.")
    return hasil.returncode


if __name__ == "__main__":
    sys.exit(main())
