"""hitung_iaa.py - kesepakatan antar-penilai untuk label tingkat kesulitan.

Masukan: master_kesulitan.csv (atau .xlsx) yang kolom reviewer_1, reviewer_2,
dan reviewer_3-nya sudah terisi easy / medium / hard.

Keluaran:
  1. Angka kesepakatan yang lazim dilaporkan di paper:
       - persentase kesepakatan penuh (ketiga penilai sama)
       - persentase kesepakatan mayoritas (minimal dua sama)
       - Fleiss' kappa            -> skala nominal
       - Krippendorff's alpha     -> skala ordinal, easy < medium < hard
     Alpha ordinal dilaporkan karena skalanya berjenjang: selisih easy-hard
     seharusnya dihitung lebih berat daripada easy-medium, dan Fleiss'
     kappa memperlakukan keduanya sama saja.
  2. Matriks kebingungan antar pasangan penilai, untuk melihat apakah
     ketidaksepakatan terpusat di satu batas (biasanya medium vs hard).
  3. daftar_adjudikasi.csv - baris yang ketiga penilainya tidak bulat, untuk
     dibahas; plus kolom final usulan berdasarkan suara mayoritas.
  4. Kalimat siap tempel untuk bagian metode.

Jalankan: python3 hitung_iaa.py [--master FILE] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASTER = HERE / "Result" / "anotasi" / "master_kesulitan.csv"
OUT = HERE / "Result" / "anotasi"

SKALA = ["easy", "medium", "hard"]
NILAI = {k: i for i, k in enumerate(SKALA)}       # jarak ordinal 0,1,2
KOLOM = ["reviewer_1", "reviewer_2", "reviewer_3"]


def muat(path: Path) -> list[dict]:
    if path.suffix.lower() == ".xlsx":
        from openpyxl import load_workbook
        ws = load_workbook(path, data_only=True).active
        baris = list(ws.values)
        kepala = [str(x) if x is not None else "" for x in baris[0]]
        return [dict(zip(kepala, ["" if v is None else str(v) for v in r]))
                for r in baris[1:]]
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fleiss_kappa(tabel: list[list[int]]) -> float:
    """tabel[i][j] = berapa penilai memberi kategori j pada unit i."""
    n_unit = len(tabel)
    if n_unit == 0:
        return float("nan")
    n_kat = len(tabel[0])
    m = sum(tabel[0])
    if m < 2:
        return float("nan")
    p_j = [sum(baris[j] for baris in tabel) / (n_unit * m) for j in range(n_kat)]
    P_i = [(sum(c * c for c in baris) - m) / (m * (m - 1)) for baris in tabel]
    P_bar = sum(P_i) / n_unit
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if P_e != 1 else float("nan")


def krippendorff_ordinal(unit_nilai: list[list[int]]) -> float:
    """Alpha dengan metrik ordinal; unit_nilai[i] = daftar kode kategori."""
    k = len(SKALA)
    # matriks koinsidensi
    o = [[0.0] * k for _ in range(k)]
    for nilai in unit_nilai:
        m_u = len(nilai)
        if m_u < 2:
            continue
        cacah = Counter(nilai)
        for c in range(k):
            for d in range(k):
                if c == d:
                    o[c][d] += cacah[c] * (cacah[c] - 1) / (m_u - 1)
                else:
                    o[c][d] += cacah[c] * cacah[d] / (m_u - 1)
    n_c = [sum(o[c]) for c in range(k)]
    n = sum(n_c)
    if n < 2:
        return float("nan")

    def delta2(c: int, d: int) -> float:
        # jarak ordinal Krippendorff: memakai cacah kategori di antaranya
        lo, hi = min(c, d), max(c, d)
        s = sum(n_c[g] for g in range(lo, hi + 1)) - (n_c[lo] + n_c[hi]) / 2
        return s * s

    Do = sum(o[c][d] * delta2(c, d) for c in range(k) for d in range(k)) / n
    De = sum(n_c[c] * n_c[d] * delta2(c, d)
             for c in range(k) for d in range(k)) / (n * (n - 1))
    return 1 - Do / De if De else float("nan")


def tafsir(x: float) -> str:
    """Rentang Landis & Koch (1977), lazim dipakai untuk membaca kappa."""
    if x != x:
        return "tidak terdefinisi"
    for batas, label in ((0.00, "buruk"), (0.20, "ringan"), (0.40, "cukup"),
                         (0.60, "sedang"), (0.80, "kuat")):
        if x < batas:
            return label if batas else "buruk"
    return "hampir sempurna"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=str(MASTER))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = muat(Path(args.master))
    print(f"{len(rows)} baris dari {Path(args.master).name}\n")

    lengkap, sebagian, salah = [], 0, []
    for r in rows:
        nilai = [(r.get(k) or "").strip().lower() for k in KOLOM]
        aneh = [v for v in nilai if v and v not in NILAI]
        if aneh:
            salah.append((r.get("id"), aneh))
            continue
        if all(nilai):
            lengkap.append((r, nilai))
        elif any(nilai):
            sebagian += 1

    if salah:
        print(f"[PERINGATAN] {len(salah)} baris memuat nilai di luar skala:")
        for qid, v in salah[:5]:
            print(f"    {qid}: {v}")
    if sebagian:
        print(f"[PERINGATAN] {sebagian} baris hanya terisi sebagian, dilewati\n")

    if not lengkap:
        print("Belum ada baris yang ketiga kolom reviewer-nya terisi.")
        print("Isi dulu reviewer_1..3 di master_kesulitan, lalu jalankan lagi.")
        print("\nSampai lembar penilaian kembali, angka kappa BELUM ADA dan")
        print("tidak boleh ditulis di manuskrip.")
        return 1

    n = len(lengkap)
    print(f"Baris dengan penilaian lengkap dari 3 penilai: {n}\n")

    # ---------- kesepakatan sederhana ----------
    bulat = sum(1 for _, v in lengkap if len(set(v)) == 1)
    mayoritas = sum(1 for _, v in lengkap if max(Counter(v).values()) >= 2)
    print(f"  kesepakatan penuh (3 sama)      : {bulat:4d}  {bulat/n*100:5.1f}%")
    print(f"  kesepakatan mayoritas (>=2 sama): {mayoritas:4d}  {mayoritas/n*100:5.1f}%")
    print(f"  tanpa mayoritas (3 beda)        : {n-mayoritas:4d}  {(n-mayoritas)/n*100:5.1f}%")

    # ---------- kappa & alpha ----------
    tabel = [[sum(1 for x in v if x == s) for s in SKALA] for _, v in lengkap]
    kappa = fleiss_kappa(tabel)
    alpha = krippendorff_ordinal([[NILAI[x] for x in v] for _, v in lengkap])
    print(f"\n  Fleiss' kappa (nominal)         : {kappa:.3f}  ({tafsir(kappa)})")
    print(f"  Krippendorff's alpha (ordinal)  : {alpha:.3f}")

    # ---------- kappa per pasangan ----------
    print("\n  Kesepakatan per pasangan penilai:")
    for a, b in combinations(range(3), 2):
        pasang = [[0] * 3 for _ in range(3)]
        for _, v in lengkap:
            pasang[NILAI[v[a]]][NILAI[v[b]]] += 1
        sama = sum(pasang[i][i] for i in range(3))
        print(f"    R{a+1} vs R{b+1}: sama {sama/n*100:5.1f}%")

    # ---------- di mana ketidaksepakatannya ----------
    batas = Counter()
    for _, v in lengkap:
        for x, y in combinations(v, 2):
            if x != y:
                batas[tuple(sorted((x, y), key=lambda s: NILAI[s]))] += 1
    if batas:
        print("\n  Letak ketidaksepakatan (pasangan label):")
        for (x, y), c in batas.most_common():
            print(f"    {x:6s} vs {y:6s} : {c:4d}")

    # ---------- sebaran per penilai ----------
    print("\n  Sebaran label tiap penilai:")
    print(f"    {'':10s} {'easy':>6s} {'medium':>7s} {'hard':>6s}")
    for i in range(3):
        c = Counter(v[i] for _, v in lengkap)
        print(f"    reviewer_{i+1} {c['easy']:6d} {c['medium']:7d} {c['hard']:6d}")

    # ---------- daftar adjudikasi ----------
    perlu = []
    for r, v in lengkap:
        if len(set(v)) == 1:
            continue
        c = Counter(v)
        top, n_top = c.most_common(1)[0]
        usul = top if n_top >= 2 else ""      # tanpa mayoritas -> harus dibahas
        perlu.append({"id": r.get("id", ""), "pertanyaan": r.get("pertanyaan", ""),
                      "reviewer_1": v[0], "reviewer_2": v[1], "reviewer_3": v[2],
                      "final_dataset": r.get("final", ""),
                      "usul_mayoritas": usul,
                      "status": "mayoritas" if usul else "tanpa mayoritas",
                      "final_adjudikasi": "", "catatan_adjudikasi": ""})
    p = out / "daftar_adjudikasi.csv"
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perlu[0].keys()) if perlu else ["id"])
        w.writeheader()
        w.writerows(perlu)
    print(f"\n  {len(perlu)} baris perlu adjudikasi -> {p.name}")

    # ---------- seberapa jauh label dataset dari hasil penilaian ----------
    cocok = sum(1 for r, v in lengkap
                if (r.get("final") or "").strip().lower() ==
                Counter(v).most_common(1)[0][0] and max(Counter(v).values()) >= 2)
    print(f"  label final dataset sama dengan suara mayoritas: "
          f"{cocok}/{n}  ({cocok/n*100:.1f}%)")

    # ---------- kalimat untuk manuskrip ----------
    kalimat = (
        f"Each item was independently rated as easy, medium, or hard by three "
        f"reviewers. On the {n} items rated by all three, the reviewers agreed "
        f"unanimously on {bulat/n*100:.1f}% of items and reached a majority on "
        f"{mayoritas/n*100:.1f}%. Inter-annotator agreement was kappa = "
        f"{kappa:.2f} (Fleiss) and alpha = {alpha:.2f} (Krippendorff, ordinal). "
        f"The {len(perlu)} items without unanimous agreement were resolved by "
        f"discussion, and the adjudicated label was taken as final."
    )
    (out / "kalimat_iaa.txt").write_text(kalimat + "\n", encoding="utf-8")
    print(f"\n{'=' * 70}\nKalimat untuk bagian metode (Result/anotasi/kalimat_iaa.txt):\n")
    print("  " + kalimat.replace(". ", ".\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
