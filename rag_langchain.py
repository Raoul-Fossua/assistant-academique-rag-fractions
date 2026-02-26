from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# =============================================================================
# rag_langchain.py
# -----------------------------------------------------------------------------
# RAG robuste + logs "jury DS" (PERF/CONFIG/CORPUS) sans casser le pipeline:
# - mêmes signatures (rag_chain(inputs)->dict)
# - mêmes env vars que ton projet
# - logs activables/désactivables par env (par défaut ON en HF)
# - logs lisibles + exploitables pour Chapitre 5 (latence + paramètres)
# =============================================================================

# ───────────────────────────── Env / Paths ─────────────────────────────
load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

# Corpus (Linux/HF sensible à la casse)
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "Corpus"))).expanduser().resolve()

# PDF / Excel (optionnels)
PDF_NAME = os.getenv("PDF_NAME", "Cours_Fractions_5e.pdf")
PDF_PATH = Path(os.getenv("PDF_PATH", str(DOCS_DIR / PDF_NAME))).expanduser().resolve()

ERREURS_XLSX = Path(os.getenv("ERREURS_XLSX", str(DOCS_DIR / "Erreurs_Fractions_5e.xlsx"))).expanduser().resolve()
REMED_XLSX = Path(os.getenv("REMED_XLSX", str(DOCS_DIR / "Remediations_Fractions_5e.xlsx"))).expanduser().resolve()

# FAISS persistant (HF: écriture autorisée dans /app)
FAISS_DIR = Path(os.getenv("FAISS_DIR", str(BASE_DIR / "faiss_store"))).expanduser().resolve()
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
FINGERPRINT_PATH = FAISS_DIR / "fingerprint.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ───────────────────────────── Logging / Perf ──────────────────────────
# Active/désactive logs perf : 1/0. Par défaut: 1 sur HF, 0 en local si tu veux.
PERF_LOG = os.getenv("PERF_LOG", "1").strip() not in {"0", "false", "False", "no", "NO"}
PERF_LOG_LEVEL = os.getenv("PERF_LOG_LEVEL", "INFO").strip().upper()  # INFO|DEBUG
PERF_LOG_JSON = os.getenv("PERF_LOG_JSON", "0").strip() in {"1", "true", "True", "yes", "YES"}

# Identifiant utile pour regrouper les logs (build/tag)
APP_VERSION = os.getenv("APP_VERSION", "").strip()  # optionnel
GIT_SHA = os.getenv("GIT_SHA", "").strip()          # optionnel

def _log(event: str, payload: Dict[str, Any]) -> None:
    """Logs stables et faciles à grep : prefix [PERF] + event."""
    if not PERF_LOG:
        return
    base = {
        "event": event,
        "ts": round(time.time(), 3),
    }
    if APP_VERSION:
        base["app_version"] = APP_VERSION
    if GIT_SHA:
        base["git_sha"] = GIT_SHA

    data = {**base, **payload}
    if PERF_LOG_JSON:
        print("[PERF] " + json.dumps(data, ensure_ascii=False))
    else:
        # Format "clé=val" bien lisible et stable
        parts = [f"{k}={data[k]}" for k in sorted(data.keys())]
        print("[PERF] " + " ".join(parts))


# ───────────────────────────── Utils ───────────────────────────────────
def _ensure_dir_writable(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    test_file = dir_path / ".write_test"
    test_file.write_text("ok", encoding="utf-8")
    test_file.unlink(missing_ok=True)


def _embeddings() -> OpenAIEmbeddings:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant (Secrets HF ou .env local).")
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)


def _llm() -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant (Secrets HF ou .env local).")
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)


def _corpus_fingerprint() -> str:
    """
    Empreinte légère (mtime + size) de tout ce qui est indexé.
    Si ça change → rebuild FAISS.
    """
    h = hashlib.sha1()
    if not DOCS_DIR.exists():
        return "no_corpus"

    files: List[Path] = []
    for p in DOCS_DIR.glob("*"):
        if p.suffix.lower() in {".txt", ".pdf", ".xlsx"}:
            files.append(p)
    for p in sorted(files, key=lambda x: x.name.lower()):
        st = p.stat()
        h.update(p.name.encode("utf-8"))
        h.update(str(st.st_mtime_ns).encode("utf-8"))
        h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()[:16]


def _corpus_stats() -> Dict[str, Any]:
    """Stats simples, utiles pour Chapitre 5 (taille corpus, nb fichiers, etc.)."""
    stats: Dict[str, Any] = {
        "docs_dir_exists": DOCS_DIR.exists(),
        "pdf_exists": PDF_PATH.exists(),
        "erreurs_xlsx_exists": ERREURS_XLSX.exists(),
        "remed_xlsx_exists": REMED_XLSX.exists(),
        "n_txt": 0,
        "n_pdf": 0,
        "n_xlsx": 0,
        "bytes_total": 0,
    }
    if not DOCS_DIR.exists():
        return stats

    for p in DOCS_DIR.glob("*"):
        suf = p.suffix.lower()
        if suf not in {".txt", ".pdf", ".xlsx"}:
            continue
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        stats["bytes_total"] += int(size)
        if suf == ".txt":
            stats["n_txt"] += 1
        elif suf == ".pdf":
            stats["n_pdf"] += 1
        else:
            stats["n_xlsx"] += 1
    return stats


# ───────────────────────────── Load corpus ─────────────────────────────
def _load_pdf_docs(pdf_path: Path) -> List[Document]:
    if not pdf_path.exists():
        return []
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()  # Document par page
    for d in docs:
        md = d.metadata or {}
        md["type"] = "pdf"
        md["source"] = str(pdf_path)
        d.metadata = md
    return docs


def _load_txt_docs(docs_dir: Path) -> List[Document]:
    if not docs_dir.exists():
        return []
    docs: List[Document] = []
    for p in sorted(docs_dir.glob("*.txt")):
        txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not txt:
            continue
        docs.append(Document(page_content=txt, metadata={"type": "txt", "source": str(p)}))
    return docs


def _load_excel_as_docs(xlsx_path: Path, default_type: str) -> List[Document]:
    if not xlsx_path.exists():
        return []
    xls = pd.ExcelFile(xlsx_path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)

    docs: List[Document] = []
    for idx, row in df.iterrows():
        parts: List[str] = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val) or str(val).strip() == "":
                continue
            parts.append(f"{col}: {val}")
        content = "\n".join(parts).strip()
        if not content:
            continue

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "type": "excel",
                    "subtype": default_type,
                    "source": str(xlsx_path),
                    "sheet": sheet,
                    "row": int(idx) + 2,
                },
            )
        )
    return docs


def _load_corpus_docs() -> List[Document]:
    # TXT recommandé HF → on le met en premier volontairement
    txt_docs = _load_txt_docs(DOCS_DIR)
    pdf_docs = _load_pdf_docs(PDF_PATH)
    err_docs = _load_excel_as_docs(ERREURS_XLSX, default_type="erreurs")
    rem_docs = _load_excel_as_docs(REMED_XLSX, default_type="remediations")
    return txt_docs + pdf_docs + err_docs + rem_docs


# ───────────────────────────── Chunking ────────────────────────────────
def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "650")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "110")),
        separators=["\nSECTION ", "\n# ", "\n## ", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


# ───────────────────────────── Vectorstore ─────────────────────────────
_VECTORSTORE: Optional[FAISS] = None


def _build_vectorstore() -> FAISS:
    t0 = time.perf_counter()

    docs = _load_corpus_docs()
    if not docs:
        raise RuntimeError(
            "❌ Aucun corpus trouvé.\n"
            f"- Attendu au moins un .txt dans: {DOCS_DIR}\n"
            f"- PDF optionnel: {PDF_PATH}\n"
        )

    t1 = time.perf_counter()
    chunks = _split_docs(docs)
    t2 = time.perf_counter()

    store = FAISS.from_documents(chunks, _embeddings())
    t3 = time.perf_counter()

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_dir_writable(FAISS_DIR)
    store.save_local(str(FAISS_DIR), index_name=FAISS_INDEX_NAME)

    fp = _corpus_fingerprint()
    FINGERPRINT_PATH.write_text(fp, encoding="utf-8")

    t4 = time.perf_counter()

    # Logs build (une fois, très utile pour Chapitre 5)
    _log(
        "vectorstore_build",
        {
            "corpus_fp": fp,
            "n_docs_raw": len(docs),
            "n_chunks": len(chunks),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "650")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "110")),
            "embed_model": os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
            "faiss_dir": str(FAISS_DIR),
            "t_load_docs_s": round(t1 - t0, 3),
            "t_chunking_s": round(t2 - t1, 3),
            "t_embed_index_s": round(t3 - t2, 3),
            "t_save_s": round(t4 - t3, 3),
            "t_total_s": round(t4 - t0, 3),
            **_corpus_stats(),
        },
    )

    return store


def _load_or_create_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    fp_now = _corpus_fingerprint()
    fp_old = None
    if FINGERPRINT_PATH.exists():
        fp_old = FINGERPRINT_PATH.read_text(encoding="utf-8", errors="ignore").strip()

    should_rebuild = (fp_old != fp_now)

    if FAISS_DIR.exists() and not should_rebuild:
        t0 = time.perf_counter()
        try:
            _VECTORSTORE = FAISS.load_local(
                str(FAISS_DIR),
                _embeddings(),
                index_name=FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            t1 = time.perf_counter()
            _log(
                "vectorstore_load",
                {
                    "corpus_fp": fp_now,
                    "faiss_dir": str(FAISS_DIR),
                    "t_load_s": round(t1 - t0, 3),
                    **_corpus_stats(),
                },
            )
            return _VECTORSTORE
        except Exception as e:
            # si chargement impossible → rebuild
            _log(
                "vectorstore_load_failed",
                {
                    "corpus_fp": fp_now,
                    "faiss_dir": str(FAISS_DIR),
                    "error": f"{type(e).__name__}: {e}",
                },
            )

    _VECTORSTORE = _build_vectorstore()
    return _VECTORSTORE


# ───────────────────────────── RAG prompt ──────────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Tu es un professeur de mathématiques niveau 5e, spécialiste des FRACTIONS.\n"
                "Tu réponds UNIQUEMENT avec le CONTEXTE fourni.\n"
                "Si le contexte ne suffit pas, dis exactement : « Je ne sais pas. »\n"
                "Format obligatoire :\n"
                "1) Idée clé (1-2 lignes)\n"
                "2) Méthode (3-6 étapes max)\n"
                "3) Exemple corrigé\n"
                "4) Piège fréquent (et correction)\n"
                "5) Mini-check (question courte)\n"
                "Style : simple, précis, pas de blabla.\n"
            ),
        ),
        ("human", "Question: {question}\n\nCONTEXTE:\n{context}"),
    ]
)


def _format_context(docs: List[Document]) -> str:
    blocks: List[str] = []
    for d in docs:
        md = d.metadata or {}
        t = md.get("type", "unknown")
        src = Path(str(md.get("source", "unknown"))).name

        if t == "pdf":
            page = md.get("page", None)
            page_str = str(page + 1) if isinstance(page, int) else (str(page) if page is not None else "?")
            header = f"[PDF {src} p.{page_str}]"
        elif t == "excel":
            sheet = md.get("sheet", "?")
            row = md.get("row", "?")
            header = f"[XLSX {src} | {sheet} | row={row}]"
        elif t == "txt":
            header = f"[TXT {src}]"
        else:
            header = f"[{t} {src}]"

        blocks.append(header + "\n" + d.page_content)

    return "\n\n---\n\n".join(blocks)


def _rank_docs(docs: List[Document]) -> List[Document]:
    # priorité: txt > pdf > excel
    order = {"txt": 0, "pdf": 1, "excel": 2}
    return sorted(docs, key=lambda d: order.get((d.metadata or {}).get("type", "unknown"), 9))


def _retriever_params() -> Dict[str, Any]:
    # On centralise pour log + stabilité
    return {
        "search_type": "mmr",
        "k": int(os.getenv("RETRIEVER_K", "5")),
        "fetch_k": int(os.getenv("RETRIEVER_FETCH_K", "25")),
        "lambda_mult": float(os.getenv("RETRIEVER_LAMBDA", "0.7")),
    }


def _context_stats(docs: List[Document], context: str) -> Dict[str, Any]:
    # Evite tokenization (plus lourd) -> stats simples
    by_type = {"txt": 0, "pdf": 0, "excel": 0, "other": 0}
    for d in docs:
        t = (d.metadata or {}).get("type", "other")
        if t not in by_type:
            t = "other"
        by_type[t] += 1
    return {
        "ctx_chars": len(context),
        "ctx_docs": len(docs),
        "ctx_txt": by_type["txt"],
        "ctx_pdf": by_type["pdf"],
        "ctx_excel": by_type["excel"],
    }


def rag_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    rag_chain({"question": "..."}) -> {"answer": str, "source_documents": List[Document]}

    Logs "Chapitre 5" (si PERF_LOG=1):
      - vectorstore_load/build (fp, tailles, chunks)
      - rag_perf (latences : load_store, retrieval, llm, total)
      - rag_config (top-k, fetch_k, mmr lambda, modèles)
    """
    question = (inputs.get("question") or "").strip()
    if not question:
        return {"answer": "Je ne sais pas.", "source_documents": []}

    # ------------------ PERF timers (exploitable jury DS) ------------------
    t0 = time.perf_counter()

    store = _load_or_create_vectorstore()
    t1 = time.perf_counter()

    rp = _retriever_params()
    retriever = store.as_retriever(
        search_type=rp["search_type"],
        search_kwargs={
            "k": rp["k"],
            "fetch_k": rp["fetch_k"],
            "lambda_mult": rp["lambda_mult"],
        },
    )

    source_docs = retriever.invoke(question)
    t2 = time.perf_counter()

    source_docs = _rank_docs(source_docs)
    context = _format_context(source_docs)

    llm = _llm()
    prompt = _RAG_PROMPT.format_messages(question=question, context=context)

    msg = llm.invoke(prompt)
    t3 = time.perf_counter()

    answer = (getattr(msg, "content", None) or "").strip() or "Je ne sais pas."

    # ------------------ Logs (propres, stables, greppables) ----------------
    # NB: on ne log pas la question complète (risque données élèves),
    # juste longueur + hash court (pour regrouper sans exposer contenu)
    q_hash = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:10]

    if PERF_LOG:
        cfg = {
            "corpus_fp": _corpus_fingerprint(),
            "embed_model": os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"),
            "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0")),
            "retriever": rp["search_type"],
            "k": rp["k"],
            "fetch_k": rp["fetch_k"],
            "lambda_mult": rp["lambda_mult"],
            "chunk_size": int(os.getenv("CHUNK_SIZE", "650")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "110")),
        }
        ctx_stats = _context_stats(source_docs, context)

        # log config 1 fois par appel (utile pour debug/rapport)
        _log("rag_config", {"q_hash": q_hash, "q_len": len(question), **cfg, **ctx_stats})

        # log perf
        _log(
            "rag_perf",
            {
                "q_hash": q_hash,
                "q_len": len(question),
                "t_load_store_s": round(t1 - t0, 3),
                "t_retrieval_s": round(t2 - t1, 3),
                "t_llm_s": round(t3 - t2, 3),
                "t_total_s": round(t3 - t0, 3),
                "n_sources": len(source_docs),
            },
        )

    return {"answer": answer, "source_documents": source_docs}