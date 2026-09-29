# bm25_index.py
# ---------------------------------------------------
# Full-text BM25 index (tanpa embeddings)
# API publik:
#   - BM25Index.build(docs)
#   - index.search(query, topn) -> List[(Document, score)]
#   - index.search_full(query, topn) -> (indices, scores ndarray)
#   - index.mmr(query, k, fetch_k, lam) -> List[(Document, bm25_score)]
#   - index.save(path, fingerprint)
#   - BM25Index.load(path)
# ---------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pickle
import sys
import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import Document

from utils import tokenize


@dataclass
class BM25Index:
    docs: List[Document]
    tokens: List[List[str]]
    bm25: BM25Okapi
    fingerprint: Optional[str] = None  # diisi saat save/load

    # ---------- Bangun index ----------
    @classmethod
    def build(cls, docs: List[Document]) -> "BM25Index":
        """
        Build BM25Okapi index dari daftar LangChain Document.
        Tokenisasi memakai utils.tokenize (regex-based).
        """
        toks = [tokenize(d.page_content) for d in docs]
        bm25 = BM25Okapi(toks)
        return cls(docs=docs, tokens=toks, bm25=bm25, fingerprint=None)

    # ---------- Query dasar ----------
    def search(self, query: str, topn: int) -> List[Tuple[Document, float]]:
        """
        Kembalikan top-n [(doc, skor)] berdasar skor BM25.
        """
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)[: max(0, topn)]
        return [(self.docs[i], float(scores[i])) for i in order]

    def search_full(self, query: str, topn: int) -> Tuple[List[int], np.ndarray]:
        """
        Kembalikan (indices_topn, semua_skor_vector).
        indices merujuk ke self.docs.
        """
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)[: max(0, topn)]
        return list(order), scores

    # ---------- MMR (diversity) di ruang leksikal ----------
    def mmr(self, query: str, k: int, fetch_k: int, lam: float) -> List[Tuple[Document, float]]:
        """
        Maximal Marginal Relevance dengan kemiripan Jaccard atas himpunan token.
        Mengembalikan hingga k item berupa [(doc, skor_bm25_asli)].
        """
        idxs, scores = self.search_full(query, fetch_k)
        if not idxs:
            return []

        # Normalisasi skor BM25 ke [0,1] untuk komponen relevansi
        sc = scores[idxs].astype(float)
        if sc.max() > sc.min():
            sc = (sc - sc.min()) / (sc.max() - sc.min())
        else:
            sc = np.zeros_like(sc)

        selected: List[int] = []
        selected_sets: List[set] = []
        candidates = list(range(len(idxs)))

        while len(selected) < min(k, len(idxs)) and candidates:
            best_c, best_val = None, -1e18
            for ci in candidates:
                doc_idx = idxs[ci]
                toks = set(self.tokens[doc_idx])
                sim = 0.0
                if selected_sets:
                    # Jaccard ke item terpilih yang paling mirip
                    sim = max(
                        (len(toks.intersection(S)) / (len(toks.union(S)) + 1e-12))
                        for S in selected_sets
                    )
                val = lam * sc[ci] - (1.0 - lam) * sim
                if val > best_val:
                    best_val, best_c = val, ci
            selected.append(best_c)  # type: ignore[arg-type]
            # type: ignore[index]
            selected_sets.append(set(self.tokens[idxs[best_c]]))
            candidates.remove(best_c)  # type: ignore[arg-type]

        return [(self.docs[idxs[i]], float(scores[idxs[i]])) for i in selected]

    # ---------- Persistensi ----------
    def save(self, path: str | Path, fingerprint: Optional[str] = None) -> None:
        """
        Simpan index ke pickle. Menyimpan docs, tokens, fingerprint.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"docs": self.docs, "tokens": self.tokens,
                "fingerprint": fingerprint or self.fingerprint}
        with open(p, "wb") as f:
            pickle.dump(data, f)
        self.fingerprint = fingerprint or self.fingerprint

    @classmethod
    def load(cls, path: str | Path) -> Optional["BM25Index"]:
        """
        Muat index dari pickle. Mengembalikan None jika gagal/berkas tak ada.
        """
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                data = pickle.load(f)
            docs: List[Document] = data["docs"]
            tokens: List[List[str]] = data["tokens"]
            fp: Optional[str] = data.get("fingerprint")
            bm25 = BM25Okapi(tokens)
            return cls(docs=docs, tokens=tokens, bm25=bm25, fingerprint=fp)
        except Exception as e:
            # Jangan diam-diam. Kegagalan di sini pernah muncul sebagai
            # "gagal memuat index" tanpa sebab, padahal akarnya MemoryError
            # saat swap habis - berjam-jam terbuang untuk mencarinya.
            print(f"[bm25_index] gagal memuat {p}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            return None


__all__ = ["BM25Index"]
