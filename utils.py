# utils.py
from __future__ import annotations
from typing import List, Set, Dict, Optional, Any, Tuple
from pathlib import Path
import re
import hashlib
import unicodedata
import numpy as np
import pandas as pd
from langchain.schema import Document

# ========= Tokenizer & Chunk ID =========
_tok_re = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _tok_re.findall(text or "")]


def chunk_id(d: Document) -> str:
    m = d.metadata or {}
    src = m.get("source", "")
    page = m.get("page", -1)
    start = m.get("start_index", None)
    h = hashlib.sha1((d.page_content[:200] or "").encode(
        "utf-8", "ignore")).hexdigest()[:8]
    return f"{src}|p{page}|s{start}|h{h}"


def preview_doc(d: Document, max_chars: int = 220) -> Tuple[str, str]:
    m = d.metadata or {}
    src = m.get("source", "doc")
    page = m.get("page", None)
    start = m.get("start_index", None)
    where = src + (f":p{page}" if page is not None else "") + \
        (f" @{start}" if start is not None else "")
    txt = d.page_content or ""
    prev = (txt[:max_chars] + "…") if len(txt) > max_chars else txt
    return where, prev

# ========= Text helpers =========


def normalize_spaces(text: str) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = (t.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2009", " ").replace("\u200A", " ")
         .replace("\u2002", " ").replace("\u2003", " ").replace("\u2004", " ").replace("\u2005", " ")
         .replace("\u2006", " ").replace("\u2007", " ").replace("\u2008", " ").replace("\u200B", ""))
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE)
    return t.strip()


def strip_codefences(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    return re.sub(r"^(```|~~~)[^\n]*\n?|(```|~~~)$", "", s, flags=re.MULTILINE).strip()


def normalize_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\.[a-z0-9]+$", "", s)  # drop extension
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    return s


# ========= Label normalization & gold parsing =========
_PAGE_RE = re.compile(r":p\d+\s*$", re.IGNORECASE)


def normalize_tag(tag: str, include_page_suffix: Optional[bool] = None) -> str:
    t = (str(tag) or "").strip().strip('"').strip("'")
    suffix = ""
    m = _PAGE_RE.search(t)
    if m:
        suffix = m.group(0)
        t = t[:m.start()].strip()
    t = t.replace("_", " ").replace("-", " ")
    t = " ".join(t.split()).lower()
    if include_page_suffix is True and suffix:
        return f"{t}{suffix.lower()}"
    if include_page_suffix is False:
        return t
    return f"{t}{suffix.lower()}" if suffix else t


def normalize_tag_list(tags: List[str], include_page_suffix: Optional[bool] = None) -> List[str]:
    return [normalize_tag(x, include_page_suffix=include_page_suffix) for x in tags]


def parse_gold_set(gold: str) -> Set[str]:
    s = (gold or "").strip().strip('"').strip("'")
    if not s:
        return set()
    return set(p.strip() for p in s.split("|") if p.strip())

# ========= Retrieval metrics (@k) =========


def _dcg(binary_relevances: List[int]) -> float:
    return float(sum(r / np.log2(i+1) for i, r in enumerate(binary_relevances, start=1)))


def rank_metrics(top_sources: List[str], gold_set: Set[str], k: int,
                 gold_mode: str = "all") -> Dict[str, float]:
    """Compute hits@k, precision@k, recall@k, f1@k, RR, AP, nDCG@k.

    gold_mode menentukan arti dari beberapa entri di gold_set:

      "all" (default) - entri saling MELENGKAPI, semuanya harus ditemukan.
          Dipakai untuk pertanyaan multi-hop dan jawaban yang terpotong
          beberapa halaman.

      "any" - entri adalah LOKASI ALTERNATIF dari jawaban yang sama, cukup
          ditemukan salah satu. Terjadi karena banyak definisi disalin
          verbatim antar peraturan UNRI. Tanpa mode ini, retrieval yang
          menemukan satu lokasi yang benar akan dihitung setengah salah:
          recall = 1/2 padahal jawabannya sudah lengkap.

    Pada mode "any" hanya kecocokan PERTAMA yang dihitung, sehingga
    R efektif = 1, recall menjadi 0 atau 1, dan AP sama dengan RR.
    """
    if k is None or k <= 0:
        k = 0
    if gold_mode not in ("all", "any"):
        raise ValueError(f"gold_mode harus 'all' atau 'any', bukan {gold_mode!r}")
    # Daftar kandidat harus berisi unit yang BERBEDA. top_sources dibangun satu
    # tag per chunk, sehingga beberapa chunk dari dokumen/halaman yang sama
    # menghasilkan tag identik. Tanpa deduplikasi, tp dapat melebihi
    # len(gold_set) sehingga recall@k, f1@k, AP, dan nDCG@k ikut melebihi 1.
    cand: List[str] = []
    if k > 0:
        seen: Set[str] = set()
        for s in top_sources:
            if s in seen:
                continue
            seen.add(s)
            cand.append(s)
            if len(cand) >= k:
                break
    rel = [1 if s in gold_set else 0 for s in cand] if cand else []
    R = int(len(gold_set))
    if gold_mode == "any" and R > 0:
        # Hanya kecocokan pertama yang dihitung; sisanya redundan, bukan tambahan.
        sudah = False
        for i, r in enumerate(rel):
            if r == 1:
                if sudah:
                    rel[i] = 0
                sudah = True
        R = 1
    tp = int(sum(rel))
    n_cand = len(cand)

    hits_k = 1.0 if tp > 0 else 0.0
    # Penyebut memakai n_cand, bukan k: setelah deduplikasi jumlah unit unik
    # yang tersedia bisa lebih kecil dari k (korpus hanya punya sedikit dokumen).
    precision_k = float(tp/n_cand) if n_cand > 0 else 0.0
    recall_k = float(tp/R) if R > 0 else float("nan")

    if R > 0:
        if precision_k == 0.0 and (recall_k == 0.0 or np.isnan(recall_k)):
            f1_k = 0.0
        else:
            if np.isnan(recall_k):
                f1_k = 0.0
            else:
                denom = precision_k + recall_k
                f1_k = float((2*precision_k*recall_k) /
                             denom) if denom > 0 else 0.0
    else:
        f1_k = float("nan")

    rr = 0.0
    for idx, r in enumerate(rel, start=1):
        if r == 1:
            rr = 1.0/idx
            break

    if R > 0:
        cum, precs = 0, []
        for idx, r in enumerate(rel, start=1):
            if r == 1:
                cum += 1
                precs.append(cum/idx)
        ap = float(sum(precs)/R) if precs else 0.0
    else:
        ap = float("nan")

    if R > 0:
        dcg_k = _dcg(rel) if rel else 0.0
        ideal = ([1]*min(R, n_cand) + [0]*max(0, n_cand-min(R, n_cand))
                 if n_cand > 0 else [])
        idcg_k = _dcg(ideal) if ideal else 0.0
        ndcg_k = float(dcg_k/idcg_k) if idcg_k > 0 else 0.0
    else:
        ndcg_k = float("nan")

    return {
        f"hits@{k}": hits_k,
        f"precision@{k}": precision_k,
        f"recall@{k}": recall_k,
        f"f1@{k}": f1_k,
        "rr": rr, "ap": ap, f"ndcg@{k}": ndcg_k,
    }


# ========= Answer metrics =========
_ID_MONTHS = {
    "januari": "01", "februari": "02", "maret": "03", "april": "04", "mei": "05", "juni": "06",
    "juli": "07", "agustus": "08", "september": "09", "oktober": "10", "november": "11", "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06", "jul": "07", "agu": "08", "agt": "08", "sept": "09", "sep": "09", "okt": "10", "nov": "11", "des": "12",
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06", "july": "07", "august": "08", "october": "10", "december": "12"
}
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _replace_month_words(text: str) -> str:
    tokens = re.split(r"(\W+)", (text or "").lower())
    return "".join(_ID_MONTHS.get(t, t) for t in tokens)


def normalize_text_for_match(s: str) -> str:
    s = _replace_month_words(s or "")
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred: str, ref: str) -> int:
    if not ref or str(ref).strip() == "":
        return 0
    return 1 if normalize_text_for_match(pred) == normalize_text_for_match(ref) else 0


def squad_f1(pred: str, ref: str) -> float:
    ptoks = normalize_text_for_match(pred).split()
    rtoks = normalize_text_for_match(ref).split()
    if not ptoks and not rtoks:
        return 1.0
    if not ptoks or not rtoks:
        return 0.0
    common, overlap = {}, 0
    for t in ptoks:
        common[t] = common.get(t, 0)+1
    for t in rtoks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap/len(ptoks)
    recall = overlap/len(rtoks)
    return 2*precision*recall/(precision+recall)


# sentence-transformers lazy load
_ST_ENCODER = None


def _get_st_encoder():
    global _ST_ENCODER
    if _ST_ENCODER is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_ENCODER = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2")
        except Exception:
            _ST_ENCODER = None
    return _ST_ENCODER


def cosine_similarity_answer(pred: str, ref: str) -> float:
    enc = _get_st_encoder()
    if enc is None or not ref or not pred:
        return float("nan")
    v = enc.encode([ref, pred], normalize_embeddings=True)
    return float(np.dot(v[0], v[1]))


def rouge_l(pred: str, ref: str) -> float:
    def lcs(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i+1][j+1] = dp[i][j]+1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        return dp[m][n]
    pred_toks = normalize_text_for_match(pred).split()
    ref_toks = normalize_text_for_match(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    L = lcs(pred_toks, ref_toks)
    prec = L/len(pred_toks)
    rec = L/len(ref_toks)
    return 0.0 if (prec+rec) == 0 else (2*prec*rec)/(prec+rec)


def compute_answer_match(pred: str, ref: str, cos_sim: Optional[float] = None,
                         use_em: bool = True, cosine_thr: float = 0.82, f1_thr: float = 0.80) -> Dict[str, float]:
    if not ref or str(ref).strip() == "":  # no reference -> undefined
        return {"em": float("nan"), "f1": float("nan"), "correct": float("nan")}
    em = exact_match(pred, ref) if use_em else 0
    f1 = squad_f1(pred, ref)
    if cos_sim is None or np.isnan(cos_sim):
        cos_sim = cosine_similarity_answer(pred, ref)
    cos_ok = (not np.isnan(cos_sim)) and (cos_sim >= cosine_thr)
    f1_ok = f1 >= f1_thr
    correct = bool(em or cos_ok or f1_ok)
    return {"em": float(em), "f1": float(f1), "correct": 1.0 if correct else 0.0}

# ========= CSV helpers =========


def read_eval_csv(file_like) -> pd.DataFrame:
    """Try ; then , with multiple encodings; robust to bad lines."""
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            if hasattr(file_like, "seek"):
                try:
                    file_like.seek(0)
                except Exception:
                    pass
            return pd.read_csv(file_like, encoding=enc, delimiter=";")
        except Exception:
            continue
    if hasattr(file_like, "seek"):
        try:
            file_like.seek(0)
        except Exception:
            pass
    return pd.read_csv(file_like, encoding="latin-1", delimiter=",", on_bad_lines="skip")


def read_csv_robust(file_like, delimiter: str = ";") -> pd.DataFrame:
    tried = []
    for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
        try:
            if hasattr(file_like, "seek"):
                try:
                    file_like.seek(0)
                except Exception:
                    pass
            return pd.read_csv(file_like, encoding=enc, delimiter=delimiter)
        except Exception as e:
            tried.append(f"{enc}: {e}")
            continue
    try:
        if hasattr(file_like, "seek"):
            try:
                file_like.seek(0)
            except Exception:
                pass
        return pd.read_csv(file_like, encoding="latin-1", delimiter=",", on_bad_lines="skip")
    except Exception as e:
        raise RuntimeError(
            f"Failed to read CSV robustly. Tried: {tried}. Last error: {e}")


def safe_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

# ========= Cache & corpus helpers =========


def cache_base_dir(base: str = "rag_cache") -> Path:
    p = Path(base).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def docs_fingerprint(docs: List[Document], *, extra: str = "", max_chars_per_doc: int = 2048) -> str:
    h = hashlib.sha256()
    if extra:
        h.update(str(extra).encode("utf-8", "ignore"))
    for d in docs or []:
        m = d.metadata or {}
        src = str(m.get("source", ""))
        page = str(m.get("page", ""))
        start = str(m.get("start_index", ""))
        body = (d.page_content or "")[:max_chars_per_doc]
        for part in (src, "|", page, "|", start, "|", body, "\n"):
            h.update(str(part).encode("utf-8", "ignore"))
    return h.hexdigest()


def summarize_corpus(docs: List[Document]) -> Dict[str, Any]:
    if not docs:
        return {"n_docs": 0, "n_sources": 0, "sources": [], "avg_chars": 0.0, "total_chars": 0}
    n = len(docs)
    sources = sorted({(d.metadata or {}).get("source", "unknown")
                     for d in docs})
    lengths = [len(d.page_content or "") for d in docs]
    total = int(sum(lengths))
    avg = total/n if n > 0 else 0.0
    return {"n_docs": n, "n_sources": len(sources), "sources": sources, "avg_chars": avg, "total_chars": total}


__all__ = [
    "tokenize", "chunk_id", "preview_doc",
    "normalize_spaces", "strip_codefences", "normalize_label",
    "normalize_tag", "normalize_tag_list", "parse_gold_set",
    "rank_metrics",
    "normalize_text_for_match", "exact_match", "squad_f1", "rouge_l",
    "cosine_similarity_answer", "compute_answer_match",
    "read_eval_csv", "read_csv_robust", "safe_to_csv",
    "cache_base_dir", "docs_fingerprint", "summarize_corpus",
]
