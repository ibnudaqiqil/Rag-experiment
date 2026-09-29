"""make_tables.py - tabel deskriptif siap cetak untuk bagian dataset di paper.

Menghasilkan dua tabel, masing-masing dalam bentuk CSV (untuk diolah lagi)
dan LaTeX booktabs (untuk langsung di-input ke manuskrip):

  Tabel 1  Deskripsi korpus  - per dokumen: jenis, tahun, halaman, chunk,
           entri gold, dan statusnya (berlaku / distractor).
  Tabel 2  Komposisi dataset - per tipe pertanyaan: cacah, sebaran kesulitan,
           median panjang jawaban, dan median tumpang tindih leksikal.

Kolom "status" dibaca dari eval_set_data_balance.DISTRAKTOR ditambah dokumen
lain yang memang tidak punya gold sama sekali, supaya pembaca paper tahu mana
yang sengaja dipakai sebagai pengecoh, bukan kelalaian anotasi.

Jalankan: python3 make_tables.py [--csv FILE] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
CSV = HERE / "evaluasi-set-507.csv"
CORPUS = HERE / "corpus"
OUT = HERE / "Result" / "tables"

# Nama pendek yang sama dengan yang dipakai di figure, supaya tabel dan
# gambar di paper bisa dirujuk silang tanpa bikin pembaca menebak.
NAMA = {
    "NOMOR 1694": ("Kepr 1694/2025 Panduan Penelitian dan PkM", "Panduan internal", 2025),
    "PANDUAN KULIAH KERJA NYATA": ("Kepr 71/2026 Panduan KKN Berdampak", "Panduan internal", 2026),
    "PANDUAN PENELITIAN DAN PENGABDIAN KEPADA MASYARAKAT.pdf":
        ("Kepr 70/2026 Panduan Penelitian dan PkM", "Panduan internal", 2026),
    "Files_2024_panduan-kukerta": ("Panduan Kukerta dan MBKM 2023", "Panduan internal", 2023),
    "Files_2026_juknis-fisip": ("Juknis FISIP Berdampak 2026", "Panduan fakultas", 2026),
    "BUKU-PANDUAN-AKADEMIK-S1-KEDOKTERAN": ("Panduan Akademik S1 Kedokteran", "Panduan fakultas", 2023),
    "PANDUAN-SMBT-UNRI-2026": ("Panduan SMBT UNRI 2026", "Panduan internal", 2026),
    "PANDUAN_PKKMB_2024": ("Panduan PKKMB UNRI 2024", "Panduan internal", 2024),
    "Panduan-registrasi-ulang": ("Panduan Registrasi Ulang", "Panduan internal", 2025),
    "Peraturan-Rektor-Tentang-Pelaksanaan-Pengabdian":
        ("Pertor Pelaksanaan Pengabdian", "Peraturan Rektor", 2021),
    "2022permendikbudristek006": ("Permendikbudristek 6/2022 (Ijazah)", "Peraturan Menteri", 2022),
    "2019permendikbud029": ("Permendikbud 29/2019 (Gratifikasi)", "Peraturan Menteri", 2019),
    "2019permendikbud030": ("Permenristekdikti 30/2019 (SSBO)", "Peraturan Menteri", 2019),
    "2017permenristekdikti039": ("Permenristekdikti 39/2017 (UKT)", "Peraturan Menteri", 2017),
    "2016permenristekdikti061": ("Permenristekdikti 61/2016 (PDDikti)", "Peraturan Menteri", 2016),
    "2016permenristekdikti005": ("Permenristekdikti 5/2016 (SSBO PTN BH)", "Peraturan Menteri", 2016),
    "2025permendiktisaintek016": ("Permendiktisaintek 16/2025 (OTK UNRI)", "Peraturan Menteri", 2025),
    "PERMENDIKBUD-RISTEK-44-2024": ("Permendikbudristek 44/2024 (Dosen)", "Peraturan Menteri", 2024),
    "Kepmendiktisaintek-358-2025-IKU": ("Kepmendiktisaintek 358/2025 (IKU)", "Keputusan Menteri", 2025),
    "TATA NASKAH DINAS": ("Permendikbud 3/2021 (Tata Naskah Dinas)", "Peraturan Menteri", 2021),
    "Kalender-Akademik-2025": ("Kalender Akademik 2025/2026", "Panduan internal", 2025),
    "pertor_3_2015": ("Pertor 3/2015 Peraturan Akademik", "Peraturan Rektor", 2015),
    "pertor_4_2021": ("Pertor 4/2021 Peraturan Akademik", "Peraturan Rektor", 2021),
    "pertor_9_2021_MBKM": ("Pertor 9/2021 (MBKM)", "Peraturan Rektor", 2021),
    "2-Penyetaraan": ("Pertor 2/2025 Penyetaraan Kredit Internasional", "Peraturan Rektor", 2025),
    "4-Penggunaan-AI": ("Pertor 4/2025 Penggunaan AI", "Peraturan Rektor", 2025),
    "5-Penerbitan-Ijazah": ("Pertor 5/2025 Penerbitan Ijazah", "Peraturan Rektor", 2025),
    "6-Pedoman-Akademik": ("Pertor 6/2025 Pedoman Akademik Magister", "Peraturan Rektor", 2025),
    "7-Integritas-Akademik": ("Pertor 7/2025 Integritas Akademik", "Peraturan Rektor", 2025),
    "8-Fast-Track": ("Pertor 8/2025 Fast Track", "Peraturan Rektor", 2025),
    "9-Perubahan-atas": ("Pertor 9/2025 Penyelenggaraan Pendidikan", "Peraturan Rektor", 2025),
}

STOP = {
    "adalah", "dalam", "untuk", "yang", "dengan", "pada", "dari", "atau", "dan",
    "oleh", "sebagai", "tidak", "dapat", "akan", "sudah", "telah", "harus",
    "paling", "setiap", "serta", "juga", "bagi", "para", "itu", "ini", "ada",
    "apa", "saja", "saya", "berapa", "bagaimana", "siapa", "kapan", "mana",
}
AKRONIM = {"sks", "ipk", "ip", "d3", "d4", "s1", "s2", "s3", "ai", "krs", "khs",
           "pkm", "iku", "ukt", "bkt", "dpl", "nidn", "otk", "tkt", "utbk", "ktm"}


def deskripsi(nama_berkas: str) -> tuple[str, str, int]:
    for kunci, nilai in NAMA.items():
        if kunci in nama_berkas:
            return nilai
    return (nama_berkas.replace(".pdf", ""), "?", 0)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", s)).strip()


def kata_isi(s: str) -> list[str]:
    return [w for w in norm(s).split()
            if w not in STOP and (w in AKRONIM or w.isdigit() or len(w) >= 4)]


def latex(header: list[str], baris: list[list[str]], rata: str, caption: str,
          label: str) -> str:
    """Tabel booktabs. Karakter khusus LaTeX di-escape seperlunya."""
    def esc(x: str) -> str:
        return (str(x).replace("&", r"\&").replace("%", r"\%")
                .replace("_", r"\_").replace("#", r"\#"))
    out = ["\\begin{table}[t]", "\\centering",
           f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
           "\\small", f"\\begin{{tabular}}{{{rata}}}", "\\toprule",
           " & ".join(esc(h) for h in header) + " \\\\", "\\midrule"]
    out += [" & ".join(esc(c) for c in r) + " \\\\" for r in baris]
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def tulis(path_dasar: Path, header: list[str], baris: list[list[str]],
          rata: str, caption: str, label: str) -> None:
    with open(path_dasar.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(baris)
    path_dasar.with_suffix(".tex").write_text(
        latex(header, baris, rata, caption, label) + "\n", encoding="utf-8")
    print(f"  {path_dasar.name}.csv + .tex   ({len(baris)} baris)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--index", default=str(HERE / "rag_cache" / "corpus31" / "bm25_index.pkl"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8"), delimiter=";"))
    print(f"{len(rows)} pertanyaan dari {Path(args.csv).name}\n")

    # ---------- data korpus ----------
    from pypdf import PdfReader
    halaman = {p.name: len(PdfReader(str(p)).pages) for p in sorted(CORPUS.glob("*.pdf"))}

    chunk: Counter = Counter()
    try:
        from bm25_index import BM25Index
        idx = BM25Index.load(args.index)
        chunk = Counter(d.metadata.get("source") for d in idx.docs)
    except Exception as e:
        print(f"  [lewat] kolom chunk butuh index: {type(e).__name__}: {e}")

    gold_entri: Counter = Counter()
    gold_hal: defaultdict[str, set] = defaultdict(set)
    for r in rows:
        for bagian in (r["gold_label"].split("|") if r["gold_label"] else []):
            nama, hal = bagian.rsplit(":p", 1)
            gold_entri[nama] += 1
            gold_hal[nama].add(int(hal))

    try:
        from eval_set_data_balance import DISTRAKTOR
    except ImportError:
        DISTRAKTOR = []

    # Dokumen yang sudah dicabut atau digantikan versi lebih baru. Ditandai
    # terpisah dari "entri gold = 0" karena dua hal itu tidak selalu sejalan:
    # Panduan Kukerta 2023 sudah digantikan tapi masih memuat dua aturan yang
    # tidak berubah, sedangkan Pertor 3/2015 dan 4/2021 masih membawa gold
    # peninggalan 40 pertanyaan versi pertama dataset.
    DIGANTIKAN = set(DISTRAKTOR) | {
        "pertor_3_2015_Peraturan-Akademik.pdf",
        "pertor_4_2021.pdf",
    }

    # ---------- Tabel 1: korpus ----------
    t1 = []
    for nama in sorted(halaman, key=lambda n: (-gold_entri[n], n)):
        judul, jenis, tahun = deskripsi(nama)
        status = "Digantikan" if nama in DIGANTIKAN else "Berlaku"
        t1.append([judul, jenis, str(tahun) if tahun else "-", str(halaman[nama]),
                   str(chunk.get(nama, 0)) if chunk else "-",
                   str(gold_entri[nama]), str(len(gold_hal[nama])), status])
    t1.append(["Total", "", "", str(sum(halaman.values())),
               str(sum(chunk.values())) if chunk else "-",
               str(sum(gold_entri.values())),
               str(sum(len(v) for v in gold_hal.values())), ""])
    tulis(out / "tabel1_korpus",
          ["Dokumen", "Jenis", "Tahun", "Halaman", "Chunk", "Entri gold",
           "Halaman gold", "Status"],
          t1, "llrrrrrl",
          "Deskripsi korpus peraturan akademik Universitas Riau. Dokumen "
          "berstatus Digantikan sudah dicabut atau diganti versi yang lebih "
          "baru; yang entri gold-nya nol berperan murni sebagai distraktor "
          "sulit karena isinya mirip dokumen yang berlaku.",
          "tab:korpus")

    # ---------- Tabel 2: komposisi dataset ----------
    tumpang: dict[str, float] = {}
    if gold_hal:
        teks = {}
        for p in sorted(CORPUS.glob("*.pdf")):
            if p.name in gold_hal:
                teks[p.name] = [norm(x.extract_text() or "")
                                for x in PdfReader(str(p)).pages]
        for r in rows:
            k = kata_isi(r["question"])
            if not k or not r["gold_label"]:
                continue
            terbaik = 0.0
            for bagian in r["gold_label"].split("|"):
                nama, hal = bagian.rsplit(":p", 1)
                t = teks[nama][int(hal)]
                terbaik = max(terbaik, sum(1 for w in k if w in t) / len(k))
            tumpang[r["id"]] = terbaik

    per = defaultdict(list)
    for r in rows:
        per[r["question_type"]].append(r)

    t2 = []
    for tipe, grup in sorted(per.items(), key=lambda x: -len(x[1])):
        sulit = Counter(g["difficulty"] for g in grup)
        panjang = [len(g["reference_answer"].split()) for g in grup]
        ov = [tumpang[g["id"]] for g in grup if g["id"] in tumpang]
        t2.append([tipe, str(len(grup)),
                   f"{len(grup)/len(rows)*100:.1f}",
                   f"{sulit['easy']}/{sulit['medium']}/{sulit['hard']}",
                   f"{median(panjang):.0f}",
                   f"{median(ov):.2f}" if ov else "-"])
    sulit = Counter(r["difficulty"] for r in rows)
    semua_ov = list(tumpang.values())
    t2.append(["Total", str(len(rows)), "100.0",
               f"{sulit['easy']}/{sulit['medium']}/{sulit['hard']}",
               f"{median([len(r['reference_answer'].split()) for r in rows]):.0f}",
               f"{median(semua_ov):.2f}" if semua_ov else "-"])
    tulis(out / "tabel2_komposisi",
          ["Tipe pertanyaan", "n", "%", "easy/medium/hard",
           "Median panjang jawaban", "Median tumpang tindih"],
          t2, "lrrcrr",
          "Komposisi dataset evaluasi. Tumpang tindih leksikal adalah bagian "
          "kata isi pertanyaan yang muncul pada halaman gold-nya; makin rendah "
          "makin sulit ditemukan pencarian berbasis kata.",
          "tab:komposisi")

    # ---------- ringkasan angka untuk badan teks ----------
    ringkas = {
        "pertanyaan": len(rows),
        "dokumen": len(halaman),
        "dokumen_bergold": sum(1 for n in halaman if gold_entri[n]),
        "dokumen_distraktor": sum(1 for n in halaman if not gold_entri[n]),
        "halaman": sum(halaman.values()),
        "halaman_gold": sum(len(v) for v in gold_hal.values()),
        "chunk": sum(chunk.values()) if chunk else None,
        "entri_gold": sum(gold_entri.values()),
        "tipe": {k: len(v) for k, v in sorted(per.items(), key=lambda x: -len(x[1]))},
        "kesulitan": dict(sulit),
        "answerable": dict(Counter(r["answerable"] for r in rows)),
        "gold_mode": dict(Counter(r["gold_mode"] for r in rows)),
        "tumpang_tindih_median": round(median(semua_ov), 3) if semua_ov else None,
    }
    (out / "ringkasan_angka.json").write_text(
        json.dumps(ringkas, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ringkasan_angka.json")
    print(f"\nSemua keluaran di {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
