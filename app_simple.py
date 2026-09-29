# app.py
# ---------------------------------------------------
# Streamlit UI — RAG Full-Text (BM25) + Simple Ablation (retrieve-only)
# Fitur:
#  - Memuat .env sekali; API keys bisa diedit & disinkronkan ke os.environ
#  - Build/Load BM25 index dari upload (PDF/TXT/MD/CSV/LOG) dengan chunking
#  - Chat RAG (jawaban dari konteks)
#  - Evaluator (retrieval & answer)
#  - Simple Ablation (Retrieve-only) + progress bar & status running test
#    Setup diuji: BM25 (Top-N), BM25+Multi-Query (RRF), BM25+HyDE,
#                 BM25+Multi-Query+Rerank, BM25+HyDE+Rerank
# ---------------------------------------------------

from __future__ import annotations
from ablation_simple import SimpleAblationConfig, run_simple_ablation  # versi progress
from evaluator import ensure_llm, generate_answer, evaluate_df
from reranker import rerank_crossencoder
from retrieval import retrieve
from bm25_index import BM25Index
from loaders import load_docs_from_uploads
from utils import cache_base_dir, docs_fingerprint, summarize_corpus, read_eval_csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ----- ensure local imports -----
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------- Page config ----------
st.set_page_config(
    page_title="RAG Full-Text • Simple Ablation + Progress", page_icon="📚", layout="wide")
st.title("📚 RAG Full-Text — Simple Ablation (BM25 • Multi-Query • HyDE • +Rerank)")

# ---------- Load .env ONCE ----------
if "env_loaded_once" not in st.session_state:
    load_dotenv()
    st.session_state["env_loaded_once"] = True

# ---------- Sidebar: .env & runtime keys ----------
st.sidebar.header("🔑 API Keys (.env loaded & editable)")
OPENAI_API_KEY = st.sidebar.text_input(
    "OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
GOOGLE_API_KEY = st.sidebar.text_input(
    "GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
DEEPSEEK_API_KEY = st.sidebar.text_input(
    "DEEPSEEK_API_KEY", type="password", value=os.getenv("DEEPSEEK_API_KEY", ""))
OPENROUTER_API_KEY = st.sidebar.text_input(
    "OPENROUTER_API_KEY", type="password", value=os.getenv("OPENROUTER_API_KEY", ""))
RERANKER_MODEL = st.sidebar.text_input("RERANKER_MODEL", value=os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

# sinkron ke runtime env
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY
if OPENROUTER_API_KEY:
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
if RERANKER_MODEL:
    os.environ["RERANKER_MODEL"] = RERANKER_MODEL

st.sidebar.caption(
    "Catatan: perubahan di sini hanya untuk runtime app ini (tidak menulis balik ke file .env).")

# ---------- Sidebar: Proyek & Index ----------
st.sidebar.header("⚙️ Index Settings")
project_name = st.sidebar.text_input("Project Name", value="default_corpus")
persist_dir = st.sidebar.text_input("Cache Folder", value="rag_cache")
overwrite = st.sidebar.checkbox("Force rebuild index", value=False)

chunk_size = st.sidebar.slider("Chunk size", 256, 3000, 800, step=64)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 600, 120, step=20)

k_docs = st.sidebar.slider("Top-k (final answer)", 1, 20, 6)
use_reranker = st.sidebar.checkbox(
    "Enable Cross-Encoder Reranker (for Chat/Eval)", value=True)

provider = st.sidebar.selectbox(
    "LLM Provider (untuk menjawab; opsional)",
    ["OpenAI", "Gemini", "DeepSeek", "OpenRouter (OpenAI-compatible)", "Non-LLM"], index=0
)
model_name = st.sidebar.text_input("Model name", value="gpt-4o-mini")

# ---------- Session ----------
for key, val in [("bm25_index", None), ("chunks", []), ("fingerprint", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ---------- Section 1: Data & Index ----------
st.header("1) Data & Index")
uploads = st.file_uploader(
    "Upload PDF/TXT/MD/CSV/LOG (multiple allowed)",
    type=["pdf", "txt", "md", "csv", "log"],
    accept_multiple_files=True
)

col1, col2, col3 = st.columns([1, 1, 1])


def _split_docs(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return splitter.split_documents(docs)


with col1:
    if st.button("🔧 Build / Reload Index"):
        with st.spinner("Building/Loading BM25 index…"):
            base = cache_base_dir(persist_dir)
            idx_dir = base / project_name
            idx_dir.mkdir(parents=True, exist_ok=True)
            cache_file = idx_dir / "bm25_index.pkl"

            raw_docs = load_docs_from_uploads(uploads) if uploads else []
            chunks = _split_docs(raw_docs, chunk_size,
                                 chunk_overlap) if raw_docs else []

            fp_extra = f"chunk_size={chunk_size}|chunk_overlap={chunk_overlap}"
            if chunks:
                fp = docs_fingerprint(chunks, extra=fp_extra)
                need_rebuild = overwrite or (
                    st.session_state["fingerprint"] != fp)

                if need_rebuild:
                    index = BM25Index.build(chunks)
                    index.save(cache_file, fingerprint=fp)
                    st.session_state["bm25_index"] = index
                    st.session_state["chunks"] = chunks
                    st.session_state["fingerprint"] = fp
                    st.success(
                        f"BM25 index built. Total chunks: {len(chunks)}")
                else:
                    index = BM25Index.load(cache_file)
                    if index is None:
                        index = BM25Index.build(chunks)
                        index.save(cache_file, fingerprint=fp)
                    st.session_state["bm25_index"] = index
                    st.session_state["chunks"] = chunks
                    st.session_state["fingerprint"] = fp
                    st.info("Corpus unchanged. Loaded index from cache.")
            else:
                index = BM25Index.load(cache_file)
                if index is None:
                    st.error(
                        "No uploads and no existing cache. Please upload documents.")
                else:
                    st.session_state["bm25_index"] = index
                    st.session_state["chunks"] = index.docs
                    st.session_state["fingerprint"] = index.fingerprint
                    st.success(
                        f"Index loaded from cache. Chunks: {len(index.docs)}")

with col2:
    st.metric("Total Chunks", len(st.session_state["chunks"]))
with col3:
    if st.session_state["chunks"]:
        summ = summarize_corpus(st.session_state["chunks"])
        st.caption(
            f"Sources: {summ['n_sources']} • Avg chars: {int(summ['avg_chars'])} • Total chars: {summ['total_chars']}")

# ---------- Section 2: Chat (RAG) ----------
st.header("2) Chat (RAG) — Answer from retrieved context")
question = st.text_input(
    "Question…", placeholder="e.g., What does Article 5 define?")

if st.button("💬 Ask"):
    if st.session_state["bm25_index"] is None:
        st.error("Index not ready. Build/Reload index first.")
    else:
        llm = ensure_llm(provider=provider,
                         model=model_name) if provider != "Non-LLM" else None
        index = st.session_state["bm25_index"]

        with st.spinner("Retrieving…"):
            docs, dbg = retrieve(
                question, index, llm=None, k_docs=k_docs, use_reranker=use_reranker, return_debug=True
            )
        with st.spinner("Generating answer…"):
            answer, tags = generate_answer(llm, question, docs)

        st.markdown("### Answer")
        st.write(answer)
        st.caption(
            f"Top-k={k_docs} • Reranker: {'ON' if use_reranker else 'OFF'}")

        st.markdown("**Sources:**")
        for i, d in enumerate(docs, 1):
            m = d.metadata or {}
            src = m.get("source", "doc")
            page = m.get("page", None)
            st.write(f"[{i}] {src}" +
                     (f", p.{page}" if page is not None else ""))

        if dbg is not None and not dbg.empty:
            with st.expander("🔍 Debug retrieval"):
                st.dataframe(dbg, use_container_width=True)

# ---------- Section 3: Evaluator ----------
st.header("3) Evaluator (Retrieval & Answer)")
st.markdown(
    "CSV minimal: **question**; opsional **reference_answer** dan **gold_label** (gunakan `File.pdf[:pN]` dipisah `|`).")

eval_csv = st.file_uploader("Evaluation CSV", type=["csv"], key="eval_csv")
eval_k = st.slider("k for @k metrics", 1, 20, max(k_docs, 5))
use_em = st.checkbox("Use Exact Match", value=True)
cos_thr = st.slider("Cosine threshold", 0.50, 0.99, 0.82, 0.01)
f1_thr = st.slider("F1 threshold", 0.10, 1.00, 0.80, 0.05)

if st.button("▶️ Run Evaluation"):
    if st.session_state["bm25_index"] is None:
        st.error("Index not ready.")
    elif eval_csv is None:
        st.error("Upload evaluation CSV first.")
    else:
        df_eval = read_eval_csv(eval_csv)
        if "question" not in df_eval.columns:
            st.error("CSV must contain column 'question'.")
        else:
            params = SimpleNamespace(k_final=eval_k)
            llm = ensure_llm(
                provider=provider, model=model_name) if provider != "Non-LLM" else None
            index = st.session_state["bm25_index"]
            reranker = (lambda q, docs, top_k: rerank_crossencoder(
                q, docs, top_k)) if use_reranker else None

            with st.spinner("Evaluating…"):
                detail, summary = evaluate_df(
                    df_eval, index, params,
                    llm=llm, reranker=reranker,
                    k_eval=eval_k, cosine_thr=cos_thr, f1_thr=f1_thr, use_em=use_em,
                    include_page_suffix=False
                )
            st.success("Done.")
            st.markdown("**Summary (rounded)**")
            st.write({k: (round(v, 4) if isinstance(v, (float, int))
                     and not pd.isna(v) else v) for k, v in summary.items()})
            st.markdown("**Per-sample Results**")
            st.dataframe(detail, use_container_width=True)
            st.download_button(
                "💾 Download evaluation (CSV)",
                data=detail.to_csv(index=False).encode("utf-8"),
                file_name="eval_results.csv", mime="text/csv"
            )

# ---------- Section 4: Simple Ablation (Retrieve-Only) ----------
st.header("4) Simple Ablation (Retrieve-Only)")
st.caption("Setup: BM25 (Top-N), BM25+Multi-Query (RRF), BM25+HyDE, BM25+Multi-Query+Rerank, BM25+HyDE+Rerank.")

k_simple = st.slider("k (top documents)", 3, 20, 10)
n_exp = st.slider("Multi-Query expansions (n)", 2, 10, 4)
run_llm_for_ablation = st.checkbox("Use LLM for Multi-Query/HyDE", value=True)

if st.button("🚀 Run Simple Ablation"):
    if st.session_state["bm25_index"] is None:
        st.error("Index not ready.")
    elif eval_csv is None:
        st.error("Upload evaluation CSV in section (3).")
    else:
        df_eval = read_eval_csv(eval_csv)
        if "question" not in df_eval.columns:
            st.error("CSV must contain column 'question'.")
        else:
            index = st.session_state["bm25_index"]
            llm_for_ablation = ensure_llm(provider=provider, model=model_name) if (
                provider != "Non-LLM" and run_llm_for_ablation) else None

            cfg = SimpleAblationConfig(
                k=k_simple, num_expansions=n_exp,
                include_page_suffix=False, normalize_labels=True
            )

            # UI progress widgets
            status = st.empty()
            prog = st.progress(0.0)
            step_info = st.empty()
            # gunakan list sebagai mutable container (menghindari nonlocal)
            last_pct = [0.0]

            # progress callback for ablation
            def _cb(done: int, total: int, setup_name: str, row_idx: int, question: str):
                pct = done / max(1, total)
                if pct - last_pct[0] >= 0.002 or done == total:  # throttle redraw
                    prog.progress(pct)
                    status.markdown(f"**Running:** {setup_name}")
                    qshort = (question[:90] +
                              "…") if len(question) > 90 else question
                    step_info.caption(
                        f"Sample {row_idx+1}/{len(df_eval)} • {qshort}")
                    last_pct[0] = pct

            with st.spinner("Running simple ablation…"):
                detail_all, summary_df = run_simple_ablation(
                    df_eval, index, config=cfg, llm=llm_for_ablation, progress_cb=_cb
                )

            prog.progress(1.0)
            status.markdown("✅ **Ablation finished.**")
            step_info.empty()

            st.markdown("### 📊 Summary")
            st.dataframe(summary_df, use_container_width=True)
            st.markdown("### 📜 Per-sample (All)")
            st.dataframe(detail_all, use_container_width=True)

st.markdown("---")
st.caption(
    "RAG full-text menggunakan BM25; opsi reranker Cross-Encoder untuk penyusunan ulang kandidat.\n"
    "Simple Ablation menghitung metrik retrieval @k saja (tanpa generasi jawaban)."
)
