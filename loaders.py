# loaders.py
# ---------------------------------------------------
# File loader robust untuk PDF/TXT/CSV/LOG/MD
# ---------------------------------------------------

import os
import tempfile
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ---------- Robust text reading ----------


def read_text_robust(path: str) -> str:
    """Coba beberapa encoding umum sebelum fallback."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


class RobustTextLoader(TextLoader):
    """Custom TextLoader dengan robust encoding detection."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        text = read_text_robust(self.file_path)
        return [Document(page_content=text, metadata={"source": os.path.basename(self.file_path)})]

# ---------- Main function ----------


def load_docs_from_uploads(uploaded_files) -> List[Document]:
    """Baca semua dokumen upload (pdf/txt/csv/md/log)."""
    docs = []
    for uf in uploaded_files or []:
        filename = uf.name
        suffix = filename.lower().split(".")[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix="." + suffix) as tmp:
            tmp.write(uf.read())
            tmp.flush()
            path = tmp.name

        if suffix == "pdf":
            loader = PyPDFLoader(path)
            sub = loader.load()
            for d in sub:
                d.metadata["source"] = filename
            docs.extend(sub)

        elif suffix in ["txt", "md", "csv", "log"]:
            loader = RobustTextLoader(path)
            sub = loader.load()
            for d in sub:
                d.metadata["source"] = filename
            docs.extend(sub)

        else:
            print(
                f"[WARN] Format {filename} belum didukung. Gunakan PDF/TXT/MD/CSV/LOG.")

    return docs


__all__ = ["read_text_robust", "RobustTextLoader", "load_docs_from_uploads"]
