"""make_figures.py - figure siap cetak untuk manuskrip, dari hasil EDA.

Setiap figure diekspor dua kali:
  .pdf  vektor, untuk disisipkan ke LaTeX/Word tanpa kehilangan ketajaman
  .png  300 dpi, untuk pratinjau dan draf

Keputusan desain:
  - Lebar 3.5 inci (satu kolom IEEE) atau 7.16 inci (dua kolom).
  - Satu seri data = satu warna. Warna kedua hanya dipakai untuk MENANDAI,
    dan selalu disertai arsiran, supaya tetap terbaca saat dicetak grayscale.
  - Grid tipis di belakang batang, sumbu tanpa kotak penuh.

Jalankan: python3 make_figures.py [--out DIR] [--font "Times New Roman"]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluasi-set-507.csv"
OVCSV = HERE / "Result" / "eda" / "eda_tumpang_tindih.csv"

BIRU = "#2a78d6"     # seri utama
ORANYE = "#eb6834"   # penanda wilayah sulit
ABU = "#b8bfc9"      # kategori bercacah kecil

# Di bawah cacah ini sebuah kategori tidak layak dipakai untuk klaim per
# kategori, jadi batangnya dibuat abu-abu sebagai peringatan.
AMBANG_KECIL = 15
GRID = "#d8dade"
TEKS = "#1a1c20"


def gaya(font: str) -> None:
    plt.rcParams.update({
        "font.family": font,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": "#9aa0a8",
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": TEKS,
        "axes.labelcolor": TEKS,
        "xtick.color": "#5a6069",
        "ytick.color": "#5a6069",
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def simpan(fig, out: Path, nama: str) -> None:
    for ext in ("pdf", "png"):
        p = out / f"{nama}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"  {nama}.pdf + .png")


def label_batang(ax, bars, nilai, fmt="{:.0f}", dx=0.01, ukuran=7) -> None:
    lim = ax.get_xlim()[1]
    for b, v in zip(bars, nilai):
        ax.text(b.get_width() + lim * dx, b.get_y() + b.get_height() / 2,
                fmt.format(v), va="center", ha="left", fontsize=ukuran, color=TEKS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "Result" / "figures"))
    ap.add_argument("--font", default="Times New Roman")
    ap.add_argument("--lang", default="both", choices=["id", "en", "both"],
                    help="bahasa label fig6 (figure ringkas untuk badan paper)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gaya(args.font)

    rows = list(csv.DictReader(open(CSV, encoding="utf-8"), delimiter=";"))
    ov = {r["id"]: float(r["tumpang_tindih"])
          for r in csv.DictReader(open(OVCSV, encoding="utf-8"))}
    print(f"Menulis figure ke {out}\n")

    # ---------- Fig 1: sebaran tumpang tindih leksikal ----------
    nilai = list(ov.values())
    tepi = [i / 10 for i in range(11)]
    cacah = [0] * 10
    for v in nilai:
        cacah[min(9, int(v * 10))] += 1

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    for i, c in enumerate(cacah):
        sulit = tepi[i] < 0.4
        ax.bar(tepi[i] + 0.05, c, width=0.092,
               color=ORANYE if sulit else BIRU,
               hatch="///" if sulit else None,
               edgecolor="white", linewidth=0.6)
    ax.set_xlabel("Tumpang tindih leksikal pertanyaan–halaman gold")
    ax.set_ylabel("Jumlah pertanyaan")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_sulit = sum(cacah[:4])
    ax.annotate(f"{n_sulit} pertanyaan ({n_sulit/len(nilai)*100:.1f}%)\nsulit secara leksikal",
                xy=(0.22, 4), xytext=(0.05, max(cacah) * 0.42),
                fontsize=7, color=ORANYE, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ORANYE, lw=0.7,
                                shrinkA=2, shrinkB=3,
                                connectionstyle="arc3,rad=-0.25"))
    med = sorted(nilai)[len(nilai) // 2]
    ax.axvline(med, color=TEKS, lw=0.8, ls=(0, (4, 2)))
    ax.text(med - 0.015, max(cacah) * 0.96, f"median {med:.2f}",
            fontsize=7, ha="right", va="top", color=TEKS)
    simpan(fig, out, "fig1_tumpang_tindih_leksikal")

    # ---------- Fig 2: komposisi ----------
    tipe = Counter(r["question_type"] for r in rows).most_common()
    sulit_ = [(k, Counter(r["difficulty"] for r in rows)[k])
              for k in ("easy", "medium", "hard")]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.4),
                                 gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.38})
    for ax, data, judul in ((a1, tipe[::-1], "(a) Tipe pertanyaan"),
                            (a2, sulit_[::-1], "(b) Tingkat kesulitan")):
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        nm = [d[0] for d in data]
        vv = [d[1] for d in data]
        warna = [ABU if v < AMBANG_KECIL else BIRU for v in vv]
        bars = ax.barh(nm, vv, color=warna, height=0.66, edgecolor="white", linewidth=0.5)
        ax.set_xlim(0, max(vv) * 1.16)
        label_batang(ax, bars, vv)
        ax.set_xlabel("Jumlah pertanyaan")
        ax.set_title(judul, loc="left", pad=6)
    # Catatan kaki menyesuaikan keadaan: selama masih ada kategori n < 15 kita
    # peringatkan pembaca, kalau sudah tidak ada kita sebutkan n terkecilnya.
    n_min = min(v for _, v in tipe)
    catatan = ("Abu-abu: n < 15, terlalu kecil untuk klaim statistik per kategori."
               if n_min < AMBANG_KECIL else
               f"Kategori terkecil n = {n_min}; seluruh tipe memenuhi n \u2265 30 "
               f"untuk uji per kategori.")
    a1.text(0.02, -0.42, catatan,
            transform=a1.transAxes, fontsize=6.8, color="#5a6069")
    simpan(fig, out, "fig2_komposisi_dataset")

    # ---------- Fig 3: entri gold per dokumen ----------
    dok = Counter()
    for r in rows:
        if not r["gold_label"]:
            dok["(tak terjawab)"] += 1
            continue
        for p in r["gold_label"].split("|"):
            dok[re.sub(r":p\d+$", "", p)] += 1
    data = dok.most_common()[::-1]

    # Nama panjang harus dipetakan eksplisit. Pemotongan buta membuat dua
    # dokumen KEPUTUSAN menghasilkan label identik, dan matplotlib menumpuk
    # keduanya pada satu posisi kategori sehingga satu batang hilang.
    KHUSUS = {
        "NOMOR 1694": "Kepr 1694/2025: Panduan Penelitian & PkM",
        "PANDUAN KULIAH KERJA NYATA": "Kepr: Panduan KKN Berdampak 2026",
        "PANDUAN PENELITIAN DAN PENGABDIAN KEPADA MASYARAKAT.pdf":
            "Kepr: Panduan Penelitian & PkM",
        "Files_2024_panduan-kukerta": "Panduan Kukerta & MBKM 2023",
        "Files_2026_juknis-fisip": "Juknis FISIP Berdampak 2026",
        "BUKU-PANDUAN-AKADEMIK-S1-KEDOKTERAN": "Panduan Akademik S1 Kedokteran",
        "PANDUAN-SMBT-UNRI-2026": "Panduan SMBT UNRI 2026",
        "PANDUAN_PKKMB_2024": "Panduan PKKMB UNRI 2024",
        "Panduan-registrasi-ulang": "Panduan Registrasi Ulang",
        "Peraturan-Rektor-Tentang-Pelaksanaan-Pengabdian":
            "Pertor: Pelaksanaan Pengabdian",
        "2022permendikbudristek006": "Permendikbudristek 6/2022 (Ijazah)",
        "2019permendikbud029": "Permendikbud 29/2019 (Gratifikasi)",
        "2019permendikbud030": "Permenristekdikti 30/2019 (SSBO)",
        "2017permenristekdikti039": "Permenristekdikti 39/2017 (UKT)",
        "2016permenristekdikti061": "Permenristekdikti 61/2016 (PDDikti)",
        "2016permenristekdikti005": "Permenristekdikti 5/2016 (SSBO PTN BH)",
    }

    def pendek(s: str) -> str:
        for kunci, nilai in KHUSUS.items():
            if kunci in s:
                return nilai
        s = s.replace(".pdf", "")
        s = re.sub(r"^(\d+)-", r"Pertor \1/2025: ", s)
        s = s.replace("Kepmendiktisaintek-358-2025-IKU", "Kepmendiktisaintek 358/2025 (IKU)")
        s = s.replace("PERMENDIKBUD-RISTEK-44-2024", "Permendikbudristek 44/2024")
        s = s.replace("2025permendiktisaintek016", "Permendiktisaintek 16/2025")
        s = s.replace("pertor_3_2015_Peraturan-Akademik", "Pertor 3/2015")
        s = s.replace("pertor_4_2021", "Pertor 4/2021")
        s = s.replace("pertor_9_2021_MBKM", "Pertor 9/2021 (MBKM)")
        s = s.replace("TATA NASKAH DINAS_Permendikbud_Nomor_3_Tahun_2021_tentang_TND_Kemdikbud_Full",
                      "Permendikbud 3/2021 (TND)")
        s = s.replace("Kalender-Akademik-2025", "Kalender Akademik 2025/26")
        s = re.sub(r"[-_]", " ", s)
        return s[:44] + "…" if len(s) > 45 else s

    def unik(labels: list[str]) -> list[str]:
        """Cegah dua dokumen berbeda memakai label yang sama."""
        lihat: dict[str, int] = {}
        hasil = []
        for x in labels:
            if x in lihat:
                lihat[x] += 1
                x = f"{x} ({lihat[x]})"
            else:
                lihat[x] = 1
            hasil.append(x)
        return hasil

    fig, ax = plt.subplots(figsize=(7.16, 3.5))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    nm = unik([pendek(d[0]) for d in data])
    vv = [d[1] for d in data]
    warna = [ABU if d[0] == "(tak terjawab)" else BIRU for d in data]
    bars = ax.barh(nm, vv, color=warna, height=0.68, edgecolor="white", linewidth=0.5)
    ax.set_xlim(0, max(vv) * 1.1)
    label_batang(ax, bars, vv)
    ax.set_xlabel("Jumlah entri gold  (pertanyaan multi-hop terhitung di dua dokumen)")
    simpan(fig, out, "fig3_gold_per_dokumen")

    # ---------- Fig 4: panjang jawaban per tipe ----------
    per = defaultdict(list)
    for r in rows:
        # Pertanyaan unanswerable memakai kalimat penolakan yang sama persis,
        # jadi panjangnya konstan dan boxplot-nya kosong. Tidak informatif.
        if r["question_type"] == "unanswerable":
            continue
        per[r["question_type"]].append(len(r["reference_answer"].split()))
    urut = sorted(per.items(), key=lambda x: len(x[1]))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    bp = ax.boxplot([v for _, v in urut], orientation="horizontal", widths=0.55,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker="o", markersize=2.2,
                                    markerfacecolor="#8d949d",
                                    markeredgecolor="none", alpha=0.7),
                    medianprops=dict(color="white", lw=1.2),
                    whiskerprops=dict(color="#7b828b", lw=0.7),
                    capprops=dict(color="#7b828b", lw=0.7))
    for kotak, (t, v) in zip(bp["boxes"], urut):
        kotak.set(facecolor=ABU if len(v) < AMBANG_KECIL else BIRU, edgecolor="none")
    ax.set_yticklabels([f"{t}  (n={len(v)})" for t, v in urut])
    ax.set_xlabel("Panjang jawaban acuan (kata)")
    n_tak = sum(1 for r in rows if r["question_type"] == "unanswerable")
    ax.text(0.0, -0.34, f"Tipe unanswerable (n={n_tak}) tidak ditampilkan: "
                        "jawabannya kalimat penolakan baku.",
            transform=ax.transAxes, fontsize=6.4, color="#5a6069")
    simpan(fig, out, "fig4_panjang_jawaban")

    # ---------- Fig 5: cakupan korpus ----------
    try:
        from pypdf import PdfReader
        hal = {p.name: len(PdfReader(str(p)).pages)
               for p in sorted((HERE / "corpus").glob("*.pdf"))}
    except Exception as e:
        print(f"  [lewat] fig5 butuh corpus/: {type(e).__name__}")
        hal = {}
    if hal:
        diuji = defaultdict(set)
        gold_dok = set()          # dokumen yang gold-nya level dokumen saja
        for r in rows:
            for p in (r["gold_label"].split("|") if r["gold_label"] else []):
                if ":p" in p:
                    nm_, h = p.rsplit(":p", 1)
                    diuji[nm_].add(int(h))
                else:
                    gold_dok.add(p)
        data = sorted(hal.items(), key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(7.16, 3.5))
        ax.set_axisbelow(True)
        ax.grid(axis="x")
        nama = unik([pendek(n) for n, _ in data])
        tot = [v for _, v in data]
        uji = [len(diuji.get(n, ())) for n, _ in data]
        # Tiga keadaan berbeda, jangan sampai terbaca sama:
        #   biru   - halaman ditandai gold
        #   arsir  - dokumen diuji, tetapi gold-nya level dokumen (tanpa nomor
        #            halaman), sehingga tidak ada halaman yang bisa diwarnai
        #   polos  - distractor murni, tidak ada pertanyaan sama sekali
        warna_dasar = ["#e7eaee"] * len(tot)
        arsir = [("///" if n in gold_dok and not diuji.get(n) else None)
                 for n, _ in data]
        ax.barh(nama, tot, color=warna_dasar, height=0.68, hatch=arsir,
                edgecolor="#b6bcc4", linewidth=0.5, label="Halaman tidak diuji")
        bars = ax.barh(nama, uji, color=BIRU, height=0.68,
                       edgecolor="white", linewidth=0.5, label="Halaman diuji (gold)")
        ax.set_xlim(0, max(tot) * 1.12)
        for b, u, t in zip(bars, uji, tot):
            ax.text(t + max(tot) * 0.012, b.get_y() + b.get_height() / 2,
                    f"{u}/{t}", va="center", ha="left", fontsize=6.8, color=TEKS)
        ax.set_xlabel("Halaman")
        from matplotlib.patches import Patch
        kunci = [Patch(facecolor=BIRU, label="Halaman ditandai gold")]
        # Entri arsir hanya ditampilkan kalau memang masih ada gold level
        # dokumen; sejak gold diseragamkan ke level halaman, tidak ada lagi.
        if any(arsir):
            kunci.append(Patch(facecolor="#e7eaee", hatch="///", edgecolor="#b6bcc4",
                               label="Diuji, tetapi gold level dokumen"))
        # Abu-abu berarti "halaman tidak ditandai gold", bukan "distractor":
        # dokumen bergold pun sebagian besar halamannya tetap abu-abu.
        kunci.append(Patch(facecolor="#e7eaee", edgecolor="#b6bcc4",
                           label="Halaman tidak ditandai gold"))
        ax.legend(handles=kunci, loc="lower right", frameon=False, fontsize=6.8)
        simpan(fig, out, "fig5_cakupan_korpus")

    # ---------- Fig 6: ringkasan satu-gambar untuk badan paper ----------
    # Kalau bagian dataset hanya boleh memuat SATU figure, dua hal inilah yang
    # perlu dilihat pembaca: apa isi datasetnya, dan seberapa sulit sebenarnya.
    # Manuskripnya berbahasa Inggris, jadi labelnya disediakan dua versi.
    L = {
        "id": dict(nama="fig6_ringkas_dataset",
                   ta="(a) Tipe pertanyaan", tb="(b) Tingkat kesulitan",
                   xa="Jumlah pertanyaan", xb="Jumlah pertanyaan",
                   tipe={}, sulit={}),
        "en": dict(nama="fig6_dataset_summary",
                   ta="(a) Question type", tb="(b) Difficulty level",
                   xa="Number of questions", xb="Number of questions",
                   tipe={"definisi": "definition", "prosedur": "procedural",
                         "komparatif": "comparative"},
                   sulit={"easy": "easy", "medium": "medium", "hard": "hard"}),
    }
    for kode in (["id", "en"] if args.lang == "both" else [args.lang]):
        fig6(out, rows, L[kode])

    print(f"\nSelesai. {len(list(out.glob('*.pdf')))} figure (.pdf + .png) di {out}")
    return 0


def fig6(out, rows, lab) -> None:
    """Figure ringkas dua panel untuk badan paper: komposisi tipe pertanyaan
    dan sebaran tingkat kesulitan.

    Panel kesulitan memakai label difficulty dari dataset. Label itu ditetapkan
    penulis dan TIDAK berkorelasi dengan kesulitan retrieval yang terukur
    (Spearman rho = -0,09; Kruskal-Wallis p = 0,13), jadi baca sebagai deskripsi
    komposisi, bukan sebagai variabel yang menjelaskan hasil retrieval.
    """
    tipe6 = sorted(Counter(r["question_type"] for r in rows).items(),
                   key=lambda x: x[1])
    cacah_sulit = Counter(r["difficulty"] for r in rows)
    urut_sulit = [k for k in ("hard", "medium", "easy") if cacah_sulit[k]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.5),
                                 gridspec_kw={"width_ratios": [1.35, 1],
                                              "wspace": 0.34})
    a1.set_axisbelow(True)
    a1.grid(axis="x")
    nm6 = [lab["tipe"].get(t, t) for t, _ in tipe6]
    vv6 = [v for _, v in tipe6]
    bars = a1.barh(nm6, vv6, color=BIRU, height=0.66,
                   edgecolor="white", linewidth=0.5)
    a1.set_xlim(0, max(vv6) * 1.18)
    label_batang(a1, bars, vv6)
    a1.set_xlabel(lab["xa"])
    a1.set_title(lab["ta"], loc="left", pad=6)

    a2.set_axisbelow(True)
    a2.grid(axis="x")
    nm_s = [lab["sulit"].get(k, k) for k in urut_sulit]
    vv_s = [cacah_sulit[k] for k in urut_sulit]
    bars2 = a2.barh(nm_s, vv_s, color=BIRU, height=0.6,
                    edgecolor="white", linewidth=0.5)
    a2.set_xlim(0, max(vv_s) * 1.2)
    label_batang(a2, bars2, vv_s)
    a2.set_xlabel(lab["xb"])
    a2.set_title(lab["tb"], loc="left", pad=6)
    simpan(fig, out, lab["nama"])


if __name__ == "__main__":
    sys.exit(main())
