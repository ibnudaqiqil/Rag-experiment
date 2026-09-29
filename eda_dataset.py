"""eda_dataset.py - analisis data eksploratif untuk evaluasi-set-507.csv.

Menjawab pertanyaan yang relevan untuk paper:

  A. Integritas   - duplikat, nilai kosong, konsistensi antar kolom.
  B. Komposisi    - sebaran per dokumen, tipe, kesulitan, granularitas gold.
  C. Teks         - panjang pertanyaan dan jawaban, kosakata.
  D. Cakupan      - berapa bagian korpus yang benar-benar diuji.
  E. Kesulitan    - tumpang tindih leksikal pertanyaan dengan halaman gold.
                    Ini penting: pertanyaan dengan tumpang tindih rendah
                    adalah tempat metode semantik (HyDE) seharusnya menang.
                    Kalau hampir semua pertanyaan tumpang tindihnya tinggi,
                    dataset ini tidak cukup menantang untuk klaim tersebut.
  F. Ambiguitas   - istilah jawaban yang tersebar di banyak dokumen.

Jalankan: python3 eda_dataset.py [--csv FILE] [--corpus DIR] [--out DIR]
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

HERE = Path(__file__).resolve().parent

STOP = {
    "adalah", "dalam", "untuk", "yang", "dengan", "pada", "dari", "atau", "dan",
    "oleh", "sebagai", "tidak", "dapat", "akan", "sudah", "telah", "harus",
    "paling", "setiap", "serta", "juga", "bagi", "para", "itu", "ini", "ada",
    "lebih", "sesuai", "ketentuan", "peraturan", "melalui", "secara", "yaitu",
    "antara", "lain", "terhadap", "maupun", "tersebut", "dimaksud", "berupa",
    "apa", "saja", "siapa", "kapan", "berapa", "bagaimana", "mana", "saya",
    "kalau", "bisa", "boleh", "apakah", "seperti", "yang", "di", "ke", "per",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Akronim domain yang pendek tapi sangat informatif. Tanpa daftar ini,
# "Berapa SKS untuk lulus D3?" kehilangan SEMUA kata kuncinya dan tumbang
# tindihnya terbaca 0.00 padahal penyebab sebenarnya beda: dokumen menulis
# "Diploma Tiga", bukan "D3".
AKRONIM = {
    "sks", "ipk", "ips", "ukt", "spp", "krs", "khs", "tpa", "loa", "iku",
    "ects", "skpi", "mbkm", "pkkmb", "uts", "uas", "do", "ai", "pa", "kkn",
    "d3", "d4", "s1", "s2", "s3", "sp1", "sp2", "unri", "nim", "nidn", "pin",
}


def isi(s: str) -> set[str]:
    """Kata isi: buang stopword, pertahankan angka dan akronim domain."""
    return {w for w in norm(s).split()
            if w not in STOP and (w.isdigit() or len(w) >= 4 or w in AKRONIM)}


def kuartil(xs: list[float]) -> tuple:
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return (0, 0, 0, 0, 0)
    def q(p):
        i = max(0, min(n - 1, int(round(p * (n - 1)))))
        return xs[i]
    return (xs[0], q(0.25), q(0.50), q(0.75), xs[-1])


def garis(judul: str) -> None:
    print(f"\n{'=' * 74}\n{judul}\n{'=' * 74}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(HERE / "evaluasi-set-507.csv"))
    ap.add_argument("--corpus", default=str(HERE / "corpus"))
    ap.add_argument("--out", default=str(HERE / "Result" / "eda"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    ringkasan: dict = {"n_baris": len(rows)}

    # ---------------- A. Integritas ----------------
    garis("A. INTEGRITAS")
    q_norm = [norm(r["question"]) for r in rows]
    dup_persis = len(rows) - len(set(q_norm))
    kosong = [r["id"] for r in rows if not r["reference_answer"].strip()]
    rusak = [r["id"] for r in rows if "�" in r["reference_answer"]]
    id_dup = len(rows) - len({r["id"] for r in rows})
    tak_konsisten = [r["id"] for r in rows
                     if (r["answerable"] == "no") != (not r["gold_label"])]
    mode_salah = [r["id"] for r in rows if r["gold_mode"] not in ("all", "any")]

    print(f"  baris                      : {len(rows)}")
    print(f"  id duplikat                : {id_dup}")
    print(f"  pertanyaan duplikat persis : {dup_persis}")
    print(f"  jawaban kosong             : {len(kosong)}")
    print(f"  karakter rusak di jawaban  : {len(rusak)}")
    print(f"  answerable vs gold cocok   : {'ya' if not tak_konsisten else 'TIDAK: ' + str(tak_konsisten[:5])}")
    print(f"  gold_mode valid            : {'ya' if not mode_salah else 'TIDAK'}")

    # near-duplicate: kesamaan Jaccard kata isi >= 0.8
    isi_q = [isi(r["question"]) for r in rows]
    mirip = []
    for i in range(len(rows)):
        if not isi_q[i]:
            continue
        for j in range(i + 1, len(rows)):
            if not isi_q[j]:
                continue
            inter = len(isi_q[i] & isi_q[j])
            if inter < 3:
                continue
            jac = inter / len(isi_q[i] | isi_q[j])
            if jac >= 0.8:
                mirip.append((round(jac, 3), rows[i]["id"], rows[j]["id"],
                              rows[i]["question"][:46], rows[j]["question"][:46]))
    print(f"  pasangan nyaris duplikat   : {len(mirip)} (Jaccard >= 0.80)")
    for j, a, b, qa, qb in sorted(mirip, reverse=True)[:6]:
        print(f"      {j:.2f}  {a} | {qa}")
        print(f"            {b} | {qb}")
    ringkasan["integritas"] = {"dup_persis": dup_persis, "nyaris_duplikat": len(mirip),
                               "jawaban_kosong": len(kosong), "karakter_rusak": len(rusak)}

    # ---------------- B. Komposisi ----------------
    garis("B. KOMPOSISI")
    for kol in ("question_type", "difficulty", "answerable", "gold_mode"):
        c = Counter(r[kol] for r in rows)
        print(f"\n  {kol}:")
        for k, v in c.most_common():
            bar = "#" * max(1, round(v / len(rows) * 50))
            print(f"    {k:14s} {v:4d}  {v/len(rows)*100:5.1f}%  {bar}")

    dok = Counter()
    for r in rows:
        if not r["gold_label"]:
            dok["(unanswerable)"] += 1
            continue
        for p in r["gold_label"].split("|"):
            dok[re.sub(r":p\d+$", "", p)] += 1
    print(f"\n  entri gold per dokumen ({len(dok)-1} dokumen bergold):")
    for k, v in dok.most_common():
        print(f"    {v:4d}  {k[:62]}")
    ringkasan["gold_per_dokumen"] = dict(dok)

    # ---------------- C. Statistik teks ----------------
    garis("C. PANJANG TEKS")
    lq = [len(r["question"].split()) for r in rows]
    la = [len(r["reference_answer"].split()) for r in rows]
    print(f"  {'':22s} {'min':>5s} {'Q1':>5s} {'median':>7s} {'Q3':>5s} {'maks':>5s} {'rata':>7s}")
    for nama, xs in (("panjang pertanyaan", lq), ("panjang jawaban", la)):
        mn, q1, md, q3, mx = kuartil(xs)
        print(f"  {nama:22s} {mn:5d} {q1:5d} {md:7d} {q3:5d} {mx:5d} {sum(xs)/len(xs):7.1f}")

    print(f"\n  panjang jawaban per tipe pertanyaan:")
    per_tipe = defaultdict(list)
    for r in rows:
        per_tipe[r["question_type"]].append(len(r["reference_answer"].split()))
    for t, xs in sorted(per_tipe.items(), key=lambda x: -len(x[1])):
        mn, q1, md, q3, mx = kuartil(xs)
        print(f"    {t:14s} n={len(xs):4d}  median {md:3d}  rentang {mn}-{mx}")

    kata = Counter()
    for r in rows:
        kata.update(isi(r["question"]))
    print(f"\n  kosakata pertanyaan: {len(kata)} kata isi berbeda")
    print(f"  10 kata tersering  : {', '.join(w for w, _ in kata.most_common(10))}")
    hapax = sum(1 for w, c in kata.items() if c == 1)
    print(f"  kata yang hanya muncul sekali: {hapax} ({hapax/len(kata)*100:.0f}%)")
    ringkasan["teks"] = {"median_pertanyaan": kuartil(lq)[2],
                         "median_jawaban": kuartil(la)[2],
                         "kosakata": len(kata), "hapax": hapax}

    # ---------------- D. Cakupan korpus ----------------
    garis("D. CAKUPAN KORPUS")
    corpus = Path(args.corpus)
    hal_dok: dict[str, int] = {}
    teks_hal: dict[tuple, str] = {}
    if corpus.exists():
        try:
            from pypdf import PdfReader
            for p in sorted(corpus.glob("*.pdf")):
                r = PdfReader(str(p))
                hal_dok[p.name] = len(r.pages)
                for i, pg in enumerate(r.pages):
                    teks_hal[(p.name, i)] = norm(pg.extract_text() or "")
        except Exception as e:
            print(f"  [PERINGATAN] gagal membaca korpus: {type(e).__name__}: {e}")
    if hal_dok:
        total_hal = sum(hal_dok.values())
        diuji = set()
        for r in rows:
            for p in (r["gold_label"].split("|") if r["gold_label"] else []):
                if ":p" in p:
                    nm, h = p.rsplit(":p", 1)
                    diuji.add((nm, int(h)))
        dok_bergold = {nm for nm, _ in diuji} | {
            re.sub(r":p\d+$", "", p) for r in rows
            for p in (r["gold_label"].split("|") if r["gold_label"] else [])}
        print(f"  dokumen di korpus          : {len(hal_dok)}")
        print(f"  dokumen punya pertanyaan   : {len(dok_bergold & set(hal_dok))}")
        print(f"  halaman di korpus          : {total_hal}")
        print(f"  halaman diuji (gold)       : {len(diuji)}  ({len(diuji)/total_hal*100:.1f}%)")
        print(f"\n  dokumen TANPA pertanyaan (murni distractor):")
        for nm in sorted(set(hal_dok) - dok_bergold):
            print(f"    {hal_dok[nm]:4d} hal  {nm[:60]}")
        ringkasan["cakupan"] = {"dokumen": len(hal_dok), "halaman": total_hal,
                                "halaman_diuji": len(diuji)}

    # ---------------- E. Kesulitan leksikal ----------------
    garis("E. TUMPANG TINDIH LEKSIKAL PERTANYAAN vs HALAMAN GOLD")
    print("  Berapa bagian kata isi pertanyaan yang muncul di halaman gold-nya.")
    print("  Rendah = BM25 sulit menemukannya, di situlah metode semantik diuji.\n")
    skor = []
    if teks_hal:
        for r in rows:
            if not r["gold_label"]:
                continue
            kq = isi(r["question"])
            if not kq:
                continue
            terbaik = 0.0
            for p in r["gold_label"].split("|"):
                if ":p" not in p:
                    continue
                nm, h = p.rsplit(":p", 1)
                t = teks_hal.get((nm, int(h)))
                if t is None:
                    continue
                terbaik = max(terbaik, sum(1 for w in kq if w in t) / len(kq))
            if terbaik or ":p" in r["gold_label"]:
                skor.append((terbaik, r))
    if skor:
        xs = [s for s, _ in skor]
        mn, q1, md, q3, mx = kuartil(xs)
        print(f"  n={len(xs)}  min {mn:.2f}  Q1 {q1:.2f}  median {md:.2f}  Q3 {q3:.2f}  maks {mx:.2f}")
        ember = Counter()
        for s in xs:
            ember[min(4, int(s * 5))] += 1
        label = ["0.0-0.2 (sangat rendah)", "0.2-0.4 (rendah)", "0.4-0.6 (sedang)",
                 "0.6-0.8 (tinggi)", "0.8-1.0 (sangat tinggi)"]
        print()
        for i, lb in enumerate(label):
            v = ember.get(i, 0)
            print(f"    {lb:26s} {v:4d}  {v/len(xs)*100:5.1f}%  {'#'*max(0,round(v/len(xs)*50))}")
        print(f"\n  15 pertanyaan dengan tumpang tindih TERENDAH (paling menantang BM25):")
        for s, r in sorted(skor, key=lambda x: x[0])[:15]:
            print(f"    {s:.2f}  {r['id']}  [{r['question_type']:10s}] {r['question'][:52]}")
        ringkasan["tumpang_tindih"] = {"median": md, "q1": q1,
                                       "rendah_<0.4": ember.get(0, 0) + ember.get(1, 0)}

    # ---------------- F. Ambiguitas lintas dokumen ----------------
    garis("F. AMBIGUITAS: ISTILAH JAWABAN TERSEBAR DI BANYAK DOKUMEN")
    if teks_hal:
        dok_teks: dict[str, str] = defaultdict(str)
        for (nm, _), t in teks_hal.items():
            dok_teks[nm] += " " + t
        sebar = []
        for r in rows:
            if not r["gold_label"]:
                continue
            ka = isi(r["reference_answer"])
            if len(ka) < 4:
                continue
            n_dok = sum(1 for nm, t in dok_teks.items()
                        if sum(1 for w in ka if w in t) / len(ka) >= 0.8)
            sebar.append((n_dok, r))
        if sebar:
            c = Counter(n for n, _ in sebar)
            print("  Berapa dokumen memuat >=80% kata isi jawaban:")
            for n in sorted(c):
                print(f"    {n:2d} dokumen : {c[n]:4d} pertanyaan")
            print(f"\n  Pertanyaan paling ambigu (jawabannya ada di banyak dokumen):")
            for n, r in sorted(sebar, key=lambda x: -x[0])[:10]:
                print(f"    {n:2d} dok  {r['id']}  {r['question'][:56]}")
            ringkasan["ambiguitas"] = {"maks_dokumen": max(c), "distribusi": dict(c)}

    # ---------------- simpan ----------------
    (out / "eda_ringkasan.json").write_text(
        json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")

    if skor:
        with open(out / "eda_tumpang_tindih.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "question", "question_type", "difficulty",
                        "gold_label", "tumpang_tindih"])
            for s, r in sorted(skor, key=lambda x: x[0]):
                w.writerow([r["id"], r["question"], r["question_type"],
                            r["difficulty"], r["gold_label"], round(s, 4)])

    print(f"\n\nDisimpan di {out}:")
    print("  eda_ringkasan.json")
    if skor:
        print("  eda_tumpang_tindih.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
