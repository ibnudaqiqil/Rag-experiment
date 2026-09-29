"""test_iaa.py - uji rumus kesepakatan di hitung_iaa.py.

Angka kappa akan masuk manuskrip, jadi rumusnya diuji terhadap kasus yang
nilainya sudah diketahui, bukan hanya "tidak error".

Jalankan: python3 test_iaa.py
"""

from __future__ import annotations

import sys

from hitung_iaa import fleiss_kappa, krippendorff_ordinal, tafsir

GAGAL = 0


def cek(nama: str, dapat, harap, toleransi: float = 1e-3) -> None:
    global GAGAL
    lolos = (abs(dapat - harap) <= toleransi if isinstance(harap, float)
             else dapat == harap)
    print(f"  {'OK  ' if lolos else 'GAGAL'} {nama}: dapat {dapat!r}, harap {harap!r}")
    if not lolos:
        GAGAL += 1


def main() -> int:
    print("Fleiss' kappa")
    # Kesepakatan sempurna -> 1.0
    cek("sepakat sempurna", fleiss_kappa([[3, 0, 0], [0, 3, 0], [0, 0, 3]]), 1.0)
    # Contoh baku Fleiss (Wikipedia): 10 unit, 14 penilai, 5 kategori -> 0.2099
    wiki = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
    cek("contoh baku Fleiss", fleiss_kappa(wiki), 0.2099, 5e-4)

    print("\nKrippendorff's alpha (ordinal)")
    cek("sepakat sempurna", krippendorff_ordinal([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), 1.0)

    # Inti skala ordinal: beda easy-medium harus dihukum lebih ringan
    # daripada beda easy-hard. Kalau tidak, alpha tidak ada gunanya di sini.
    dekat = [[0, 0, 1], [1, 1, 2], [0, 1, 1], [1, 2, 2], [0, 0, 1], [1, 1, 2]]
    jauh = [[0, 0, 2], [0, 2, 2], [0, 0, 2], [0, 2, 2], [0, 0, 2], [0, 2, 2]]
    a_dekat, a_jauh = krippendorff_ordinal(dekat), krippendorff_ordinal(jauh)
    print(f"  info  beda bersebelahan alpha={a_dekat:.3f}, "
          f"beda berjauhan alpha={a_jauh:.3f}")
    cek("beda bersebelahan lebih ringan daripada berjauhan", a_dekat > a_jauh, True)

    print("\nTafsir Landis & Koch")
    for nilai, harap in ((-0.10, "buruk"), (0.05, "ringan"), (0.30, "cukup"),
                         (0.50, "sedang"), (0.70, "kuat"), (0.85, "hampir sempurna")):
        cek(f"kappa {nilai}", tafsir(nilai), harap)

    print(f"\n{'SEMUA LOLOS' if not GAGAL else f'{GAGAL} UJI GAGAL'}")
    return 1 if GAGAL else 0


if __name__ == "__main__":
    sys.exit(main())
