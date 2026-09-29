# test_rank_metrics.py
# ---------------------------------------------------------------
# Uji regresi untuk bug: tag duplikat pada top_sources membuat
# recall@k, f1@k, AP, dan nDCG@k melebihi 1.
#
# Jalankan: ./.venv/bin/python test_rank_metrics.py
# ---------------------------------------------------------------

import itertools
import random
import sys

from utils import rank_metrics

BOUNDED = ("precision@", "recall@", "f1@", "ndcg@")
gagal = []


def periksa(nama, top, gold, k):
    m = rank_metrics(top, gold, k)
    for key, val in m.items():
        if val != val:          # NaN dilewati (kasus gold kosong)
            continue
        if key.startswith(BOUNDED) or key in ("rr", "ap", f"hits@{k}"):
            if not (0.0 <= val <= 1.0 + 1e-9):
                gagal.append(f"{nama}: {key} = {val:.4f} di luar [0,1]")
    return m


# --- Kasus yang dulu rusak: semua chunk teratas dari dokumen gold yang sama ---
print("Kasus duplikasi tag (gold level dokumen):")
for k in (2, 3, 5, 7, 10, 15):
    m = periksa(f"semua-duplikat-k{k}", ["a.pdf"] * k, {"a.pdf"}, k)
    print(f"  k={k:2d}  precision={m[f'precision@{k}']:.3f} "
          f"recall={m[f'recall@{k}']:.3f} ap={m['ap']:.3f} "
          f"ndcg={m[f'ndcg@{k}']:.3f}")

# --- Perilaku yang benar tetap dipertahankan ---
print("\nKasus normal:")
m = periksa("tepat-di-peringkat-1", ["a.pdf", "b.pdf", "c.pdf"], {"a.pdf"}, 3)
assert m["rr"] == 1.0 and m["ap"] == 1.0, m
print(f"  gold ketemu di peringkat 1 -> rr={m['rr']:.2f} ap={m['ap']:.2f}")

m = periksa("tepat-di-peringkat-3", ["x.pdf", "y.pdf", "a.pdf"], {"a.pdf"}, 3)
assert abs(m["rr"] - 1/3) < 1e-9, m
print(f"  gold ketemu di peringkat 3 -> rr={m['rr']:.3f} ap={m['ap']:.3f}")

m = periksa("tidak-ketemu", ["x.pdf", "y.pdf"], {"a.pdf"}, 2)
assert m["hits@2"] == 0.0 and m["rr"] == 0.0, m
print(f"  gold tidak ketemu        -> hits={m['hits@2']:.0f} ap={m['ap']:.2f}")

m = periksa("dua-gold-duplikat", ["a.pdf", "a.pdf", "b.pdf", "b.pdf", "z.pdf"],
            {"a.pdf", "b.pdf"}, 5)
print(f"  2 gold, tiap gold muncul 2x -> recall={m['recall@5']:.2f} ap={m['ap']:.2f}")
assert m["recall@5"] == 1.0, m

# --- Uji acak menyeluruh ---
print("\nUji acak 20.000 kombinasi:")
random.seed(0)
docs = ["a.pdf", "b.pdf", "c.pdf", "d.pdf"]
for i in range(20000):
    k = random.randint(1, 12)
    top = [random.choice(docs) for _ in range(random.randint(1, 15))]
    gold = set(random.sample(docs, random.randint(1, 3)))
    periksa(f"acak#{i}", top, gold, k)

if gagal:
    print(f"\nGAGAL: {len(gagal)} pelanggaran batas [0,1]")
    for g in gagal[:10]:
        print("   ", g)
    sys.exit(1)
print("\nOK: semua metrik berada di dalam [0,1].")


# =================================================================
# Verifikasi khusus nDCG@k
# =================================================================
import math

print("\n" + "=" * 58)
print("Verifikasi nDCG@k")
print("=" * 58)


def ndcg_referensi(top, gold, k):
    """Implementasi pembanding yang ditulis terpisah dari utils.py."""
    seen, cand = set(), []
    for s in top:
        if s not in seen:
            seen.add(s)
            cand.append(s)
        if len(cand) >= k:
            break
    rel = [1 if s in gold else 0 for s in cand]
    R = len(gold)
    if R == 0 or not cand:
        return float("nan") if R == 0 else 0.0
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, 1))
    n_ideal = min(R, len(cand))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --- Nilai yang dihitung tangan ---
print("\nNilai yang dihitung manual:")
kasus = [
    (["a.pdf", "x.pdf", "y.pdf"], {"a.pdf"}, 3, 1.0,            "gold di peringkat 1"),
    (["x.pdf", "a.pdf", "y.pdf"], {"a.pdf"}, 3, 1/math.log2(3), "gold di peringkat 2"),
    (["x.pdf", "y.pdf", "a.pdf"], {"a.pdf"}, 3, 1/math.log2(4), "gold di peringkat 3"),
    (["a.pdf", "b.pdf", "y.pdf"], {"a.pdf", "b.pdf"}, 3, 1.0,   "2 gold di peringkat 1-2"),
    (["a.pdf", "y.pdf", "b.pdf"], {"a.pdf", "b.pdf"}, 3,
     (1 + 1/math.log2(4)) / (1 + 1/math.log2(3)),               "2 gold di peringkat 1 dan 3"),
    (["x.pdf", "y.pdf", "z.pdf"], {"a.pdf"}, 3, 0.0,            "gold tidak ketemu"),
]
for top, gold, k, harapan, label in kasus:
    got = rank_metrics(top, gold, k)[f"ndcg@{k}"]
    ok = abs(got - harapan) < 1e-9
    print(f"  {'OK ' if ok else 'SALAH'} {label:30s} nDCG={got:.4f} (harap {harapan:.4f})")
    if not ok:
        gagal.append(f"nDCG manual '{label}': {got} != {harapan}")

# --- Monotonisitas: gold yang lebih tinggi peringkatnya tidak boleh skor lebih rendah ---
print("\nMonotonisitas terhadap peringkat (1 gold, k=8):")
sebelum = None
for pos in range(8):
    top = ["x%d.pdf" % i for i in range(8)]
    top[pos] = "a.pdf"
    nd = rank_metrics(top, {"a.pdf"}, 8)["ndcg@8"]
    print(f"  gold di peringkat {pos+1}: nDCG={nd:.4f}")
    if sebelum is not None and nd > sebelum + 1e-12:
        gagal.append(f"nDCG naik saat peringkat memburuk: {sebelum} -> {nd}")
    sebelum = nd

# --- Cocokkan dengan implementasi pembanding pada 20.000 kasus acak ---
print("\nCocokkan dengan implementasi pembanding (20.000 kasus acak):")
random.seed(11)
docs = ["a.pdf", "b.pdf", "c.pdf", "d.pdf", "e.pdf"]
beda = 0
for _ in range(20000):
    k = random.randint(1, 12)
    top = [random.choice(docs) for _ in range(random.randint(1, 15))]
    gold = set(random.sample(docs, random.randint(1, 3)))
    a = rank_metrics(top, gold, k)[f"ndcg@{k}"]
    b = ndcg_referensi(top, gold, k)
    if abs(a - b) > 1e-9:
        beda += 1
        if beda <= 3:
            gagal.append(f"nDCG beda: utils={a:.6f} referensi={b:.6f} top={top} gold={gold} k={k}")
print(f"  selisih > 1e-9: {beda} dari 20000")

if gagal:
    print(f"\nGAGAL: {len(gagal)} masalah")
    for g in gagal[:10]:
        print("   ", g)
    sys.exit(1)
print("\nOK: nDCG@k terverifikasi.")


# =================================================================
# gold_mode="any": entri gold adalah lokasi alternatif
# =================================================================
print("\n" + "=" * 58)
print("Verifikasi gold_mode")
print("=" * 58)

GOLD2 = {"a.pdf:p7", "b.pdf:p6"}          # definisi sama di dua dokumen
TOP = ["a.pdf:p7", "x.pdf:p1", "x.pdf:p2", "x.pdf:p3", "x.pdf:p4"]

m_all = periksa("dua-gold-mode-all", TOP, GOLD2, 5)
m_any = rank_metrics(TOP, GOLD2, 5, gold_mode="any")
for key, val in m_any.items():
    if val == val and not (0.0 <= val <= 1.0 + 1e-9):
        gagal.append(f"mode-any: {key} = {val} di luar [0,1]")

print("\nSatu dari dua lokasi alternatif ditemukan, jawaban BENAR:")
print(f"  {'metrik':12s} {'all':>8s} {'any':>8s}")
for key in ("hits@5", "precision@5", "recall@5", "f1@5", "rr", "ap", "ndcg@5"):
    print(f"  {key:12s} {m_all[key]:8.3f} {m_any[key]:8.3f}")

assert m_any["recall@5"] == 1.0, m_any
assert abs(m_any["ap"] - m_any["rr"]) < 1e-9, "mode any: AP harus sama dengan RR"
assert m_any["ndcg@5"] == 1.0, m_any
assert m_all["recall@5"] == 0.5, m_all

# gold tunggal: kedua mode harus identik
g1 = {"a.pdf:p7"}
for k in (1, 3, 5):
    a = rank_metrics(TOP, g1, k)
    b = rank_metrics(TOP, g1, k, gold_mode="any")
    if a != b:
        gagal.append(f"gold tunggal k={k}: mode all dan any berbeda")
print("\ngold tunggal -> mode 'all' dan 'any' identik: OK")

# keduanya ditemukan: mode any tidak boleh melebihi 1
m2 = rank_metrics(["a.pdf:p7", "b.pdf:p6", "x.pdf:p1"], GOLD2, 3, gold_mode="any")
print(f"kedua lokasi ditemukan -> recall={m2['recall@3']:.2f} ap={m2['ap']:.2f} "
      f"precision={m2['precision@3']:.2f}")
assert m2["recall@3"] == 1.0 and m2["ap"] == 1.0, m2

try:
    rank_metrics(TOP, GOLD2, 5, gold_mode="salah")
    gagal.append("gold_mode tidak valid seharusnya ditolak")
except ValueError:
    print("gold_mode tidak valid ditolak: OK")

if gagal:
    print(f"\nGAGAL: {len(gagal)} masalah")
    for g in gagal[:10]:
        print("   ", g)
    sys.exit(1)
print("\nOK: gold_mode terverifikasi.")
