"""llm_cache.py - menyimpan dan memakai ulang keluaran LLM.

Dua gunanya:

1. Reprodusibilitas. Seluruh permintaan dan jawaban LLM tersimpan di
   llm_cache/responses.jsonl sehingga hasil eksperimen dapat ditelusuri dan
   dijalankan ulang tanpa memanggil API lagi.

2. Biaya. Ekspansi Multi-Query dan HyDE untuk satu pertanyaan tidak
   bergantung pada nilai k maupun aktif/tidaknya reranker. Tanpa cache,
   ekspansi yang sama dipanggil ulang di setiap konfigurasi sweep.

Format JSONL, satu baris per pemanggilan unik:
  {"key", "kind", "provider", "model", "query", "n", "prompt",
   "response", "latency_s", "ts"}
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", HERE / "llm_cache"))
CACHE_FILE = CACHE_DIR / "responses.jsonl"

_lock = threading.Lock()
_mem: dict[str, str] = {}
_loaded = False
_stats = {"hit": 0, "miss": 0, "error": 0, "tunggu_s": 0.0}

# Gemini tier gratis membatasi 15 permintaan/menit. Tanpa jeda, sebagian
# panggilan gagal dengan ResourceExhausted dan datanya hilang dari eksperimen.
RPM = int(os.getenv("LLM_RPM", "180"))
_interval = 60.0 / max(1, RPM)
_terakhir = [0.0]


def _tahan_laju() -> None:
    with _lock:
        jeda = _interval - (time.time() - _terakhir[0])
        if jeda > 0:
            _stats["tunggu_s"] += jeda
        _terakhir[0] = time.time() + max(0.0, jeda)
    if jeda > 0:
        time.sleep(jeda)


def _key(kind: str, provider: str, model: str, query: str, n: int) -> str:
    bahan = f"{kind}|{provider}|{model}|{n}|{query.strip()}"
    return hashlib.sha256(bahan.encode("utf-8")).hexdigest()[:32]


def _muat() -> None:
    global _loaded
    if _loaded:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        rusak = 0
        with open(CACHE_FILE, encoding="utf-8") as f:
            for baris in f:
                baris = baris.strip()
                if not baris:
                    continue
                try:
                    rec = json.loads(baris)
                    _mem[rec["key"]] = rec["response"]
                except Exception:
                    rusak += 1
        if rusak:
            print(f"[llm_cache] {rusak} baris cache rusak dilewati")
    _loaded = True


def ambil_atau_panggil(kind: str, provider: str, model: str, query: str,
                       n: int, prompt: str, panggil) -> Optional[str]:
    """Kembalikan respons dari cache, atau panggil `panggil()` lalu simpan.

    `panggil` adalah fungsi tanpa argumen yang mengembalikan teks respons.
    Mengembalikan None bila pemanggilan gagal (pemanggil harus menanganinya).
    """
    _muat()
    k = _key(kind, provider, model, query, n)
    with _lock:
        if k in _mem:
            _stats["hit"] += 1
            return _mem[k]

    _tahan_laju()
    t0 = time.perf_counter()
    try:
        resp = panggil()
    except Exception as e:
        _stats["error"] += 1
        print(f"[llm_cache] GAGAL {kind} ({type(e).__name__}): {str(e)[:160]}")
        return None
    lat = time.perf_counter() - t0

    rec = {
        "key": k, "kind": kind, "provider": provider, "model": model,
        "query": query, "n": n, "prompt": prompt, "response": resp,
        "latency_s": round(lat, 3),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with _lock:
        _mem[k] = resp
        _stats["miss"] += 1
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return resp


def statistik() -> dict:
    _muat()
    return {**_stats, "tersimpan": len(_mem)}


def ringkas() -> str:
    s = statistik()
    return (f"cache LLM: {s['hit']} hit, {s['miss']} panggilan baru, "
            f"{s['error']} gagal, {s['tersimpan']} entri, "
            f"tunggu {s['tunggu_s']/60:.1f} menit")
