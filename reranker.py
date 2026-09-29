# reranker.py
# ---------------------------------------------------
# Reranker (opsional) dengan fallback aman.
# Mendukung Cross-Encoder (sentence-transformers) dan monoT5 (castorini/monot5-*).
# API publik yang dipakai retrieval.py:
#   - rerank_crossencoder(query, docs, top_k, model_name=None, max_chars=1024)
#     (nama tetap demi backward compatibility, namun dapat memilih monoT5 via RERANKER_MODEL)
# ---------------------------------------------------

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import os
import logging

from langchain.schema import Document

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Coba import dependensi opsional
try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# transformers untuk monoT5 (opsional)
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None

# Masked LM for TILDE/TILDEv2-style term scoring (opsional)
try:
    from transformers import AutoModelForMaskedLM, BertTokenizerFast
except Exception:
    AutoModelForMaskedLM = None
    BertTokenizerFast = None


# ---------- Util perangkat ----------
def _pick_device() -> str:
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _doc_text(d: Document, max_chars: int) -> str:
    t = (d.page_content or "")[:max_chars]
    return t if t.strip() else " "


# ---------- No-Op Reranker ----------
class _NoOpReranker:
    name = "noop"

    def __call__(self, query: str, docs: List[Document], top_k: int) -> Tuple[List[Document], Dict[str, float]]:
        top_k = max(0, min(top_k, len(docs)))
        # kembalikan urutan apa adanya
        return docs[:top_k], {id(d): 0.0 for d in docs}


# ---------- Cross-Encoder Reranker ----------
class _CrossEncoderReranker:
    def __init__(self, model_name: str, device: Optional[str] = None, max_chars: int = 1024):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers tidak tersedia.")
        self.model_name = model_name
        self.device = device or _pick_device()
        self.max_chars = max_chars
        self.model = CrossEncoder(model_name, device=self.device)

    def __call__(self, query: str, docs: List[Document], top_k: int) -> Tuple[List[Document], Dict[str, float]]:
        if not docs:
            return [], {}
        pairs = [(query, _doc_text(d, self.max_chars)) for d in docs]
        try:
            scores = self.model.predict(
                pairs, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            logger.warning(
                f"CrossEncoder.predict gagal: {e}. Fallback ke NoOp.")
            return _NoOpReranker()(query, docs, top_k)
        order = scores.argsort()[::-1]
        top_k = max(0, min(top_k, len(docs)))
        ordered_docs = [docs[i] for i in order[:top_k]]
        score_map = {id(docs[i]): float(scores[i]) for i in order}
        return ordered_docs, score_map


# ---------- monoT5 Reranker ----------
class _MonoT5Reranker:
    """
    Reranker berbasis generatif T5 (monoT5). Menghitung skor relevansi
    dengan log-likelihood output "true" vs "false".

    Model contoh: "castorini/monot5-base-msmarco" atau "castorini/monot5-large-msmarco".
    """

    def __init__(self, model_name: str, device: Optional[str] = None, max_chars: int = 1024):
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
            raise RuntimeError("transformers/torch tidak tersedia untuk monoT5.")
        self.model_name = model_name
        self.device = device or _pick_device()
        self.max_chars = max_chars
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Pre-tokenize target label ids
        self.true_ids = self.tokenizer("true", return_tensors="pt").input_ids
        self.false_ids = self.tokenizer("false", return_tensors="pt").input_ids
        if self.true_ids.device.type != self.device:
            self.true_ids = self.true_ids.to(self.device)
        if self.false_ids.device.type != self.device:
            self.false_ids = self.false_ids.to(self.device)

    def _format_input(self, query: str, doc_text: str) -> str:
        return f"Query: {query}\nDocument: {doc_text}\nRelevant:"

    @torch.inference_mode()
    def _score_pair(self, q: str, d_text: str) -> float:
        text = self._format_input(q, d_text)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)

        # Compute losses for targets "true" and "false"
        out_true = self.model(input_ids=input_ids, attention_mask=attn, labels=self.true_ids)
        out_false = self.model(input_ids=input_ids, attention_mask=attn, labels=self.false_ids)
        # Higher is better: use negative loss; or margin
        s_true = -float(out_true.loss.item())
        s_false = -float(out_false.loss.item())
        return s_true - s_false

    def __call__(self, query: str, docs: List[Document], top_k: int) -> Tuple[List[Document], Dict[str, float]]:
        if not docs:
            return [], {}
        scores = []
        for d in docs:
            d_text = _doc_text(d, self.max_chars)
            try:
                sc = self._score_pair(query, d_text)
            except Exception as e:
                logger.warning(f"monoT5 scoring gagal untuk satu dokumen: {e}")
                sc = float("-inf")
            scores.append(sc)

        order = list(sorted(range(len(docs)), key=lambda i: scores[i], reverse=True))
        top_k = max(0, min(top_k, len(docs)))
        ordered_docs = [docs[i] for i in order[:top_k]]
        score_map = {id(docs[i]): float(scores[i]) for i in order}
        return ordered_docs, score_map


# ---------- TILDEv2-like Reranker ----------
class _TILDEv2Reranker:
    """
    Reranker pendekatan TILDE/TILDEv2: menilai kecocokan berbasis
    likelihood token independen dari query terhadap token dokumen.

    Implementasi ringan (approximation):
    - Hitung bobot per token query via masked-LM likelihood.
    - Skor dokumen = jumlah bobot token query yang muncul di dokumen.

    Model contoh: "ielab/TILDE" (HF). Jika Anda memiliki TILDEv2 di HF,
    set RERANKER_MODEL ke nama yang sesuai.
    """

    def __init__(self, model_name: str, device: Optional[str] = None, max_chars: int = 1024):
        if AutoModelForMaskedLM is None or torch is None:
            raise RuntimeError("transformers/torch tidak tersedia untuk TILDEv2.")
        # Prefer tokenizer fast untuk konsistensi tokenization
        self.tokenizer = None
        if BertTokenizerFast is not None:
            try:
                self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
            except Exception:
                self.tokenizer = None
        if self.tokenizer is None:
            # fallback ke AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.device = device or _pick_device()
        self.model.to(self.device)
        self.model.eval()
        self.model_name = model_name
        self.max_chars = max_chars

        # Token id untuk [MASK]
        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise RuntimeError("Model tidak mendukung masked-LM (mask_token_id None).")

    def _query_token_weights(self, query: str) -> Dict[int, float]:
        """Estimasi bobot token query dengan masked-LM likelihood.
        Untuk tiap token t di query, mask token tsb dan ambil logit(k) untuk token aslinya.
        """
        enc = self.tokenizer(query, add_special_tokens=True, return_tensors="pt")
        input_ids = enc.input_ids.to(self.device)
        attn = enc.attention_mask.to(self.device)

        token_ids = input_ids[0].tolist()  # 1D
        weights: Dict[int, float] = {}
        # Iterasi posisi token non-special
        special_ids = set([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id, self.tokenizer.unk_token_id])
        for pos, tok_id in enumerate(token_ids):
            if tok_id in special_ids or tok_id == self.mask_token_id:
                continue
            masked = input_ids.clone()
            masked[0, pos] = self.mask_token_id
            with torch.inference_mode():
                logits = self.model(masked, attention_mask=attn).logits
                # logit pada posisi [MASK]
                mask_logits = logits[0, pos]
                # gunakan nilai logit token asli sebagai bobot
                w = float(mask_logits[tok_id].item())
            # akumulasi (kalau token berulang), ambil max
            if tok_id in weights:
                weights[tok_id] = max(weights[tok_id], w)
            else:
                weights[tok_id] = w
        return weights

    def _doc_token_set(self, text: str) -> set[int]:
        enc = self.tokenizer(text, add_special_tokens=True, return_tensors="pt", truncation=True)
        ids = enc.input_ids[0].tolist()
        special_ids = set([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id, self.tokenizer.unk_token_id, self.mask_token_id])
        return set(i for i in ids if i not in special_ids)

    def __call__(self, query: str, docs: List[Document], top_k: int) -> Tuple[List[Document], Dict[str, float]]:
        if not docs:
            return [], {}
        try:
            q_weights = self._query_token_weights(query)
        except Exception as e:
            logger.warning(f"TILDEv2 query weighting gagal: {e}. Fallback NoOp.")
            return _NoOpReranker()(query, docs, top_k)

        scores = []
        for d in docs:
            text = _doc_text(d, self.max_chars)
            try:
                tok_set = self._doc_token_set(text)
                sc = sum(w for tid, w in q_weights.items() if tid in tok_set)
            except Exception as e:
                logger.warning(f"TILDEv2 doc scoring gagal: {e}")
                sc = float('-inf')
            scores.append(sc)

        order = list(sorted(range(len(docs)), key=lambda i: scores[i], reverse=True))
        top_k = max(0, min(top_k, len(docs)))
        ordered_docs = [docs[i] for i in order[:top_k]]
        score_map = {id(docs[i]): float(scores[i]) for i in order}
        return ordered_docs, score_map
# ---------- Lazy factory & singletons ----------
_RERANKER_SINGLETON: Optional[object] = None
_RERANKER_NAME: Optional[str] = None


def _get_reranker(model_name: Optional[str], max_chars: int) -> object:
    """
    Lazy-load reranker. Prioritas model:
      argumen -> env RERANKER_MODEL -> default 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    Jika gagal, kembalikan NoOp.
    """
    global _RERANKER_SINGLETON, _RERANKER_NAME
    wanted = (model_name or os.getenv("RERANKER_MODEL")
              or "cross-encoder/ms-marco-MiniLM-L-6-v2").strip()
    if _RERANKER_SINGLETON is not None and _RERANKER_NAME == wanted:
        return _RERANKER_SINGLETON
    try:
        # Heuristik pemilihan backend berdasarkan nama model
        lower = wanted.lower()
        if ("monot5" in lower) or (lower.startswith("castorini/monot5")):
            rr = _MonoT5Reranker(wanted, device=None, max_chars=max_chars)
        elif ("tilde" in lower):
            rr = _TILDEv2Reranker(wanted, device=None, max_chars=max_chars)
        else:
            rr = _CrossEncoderReranker(wanted, device=None, max_chars=max_chars)
        _RERANKER_SINGLETON, _RERANKER_NAME = rr, wanted
        dev = getattr(rr, "device", "cpu")
        logger.info(f"[reranker] Loaded {wanted} on device={dev}")
        return rr
    except Exception as e:
        logger.warning(f"[reranker] Gagal init '{wanted}': {e}. Pakai NoOp.")
        _RERANKER_SINGLETON, _RERANKER_NAME = _NoOpReranker(), "noop"
        return _RERANKER_SINGLETON


# ---------- API yang dipakai retrieval.py ----------
def rerank_crossencoder(
    query: str,
    docs: List[Document],
    top_k: int,
    model_name: Optional[str] = None,
    max_chars: int = 1024
) -> Tuple[List[Document], Dict[str, float]]:
    """
    Rerank list Document dengan backend yang tersedia:
    - Cross-Encoder (default)
    - monoT5 (jika RERANKER_MODEL mengarah ke model monot5)
    Fallback aman: NoOp (tidak mengubah urutan).
    """
    rer = _get_reranker(model_name, max_chars=max_chars)
    return rer(query, docs, top_k)


__all__ = ["rerank_crossencoder"]
