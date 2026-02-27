from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# rag_langchain.py — Version robuste (anti-crash proxies/httpx/openai)
# -----------------------------------------------------------------------------
# Objectif:
# - Ne PAS crasher si certaines dépendances ne sont pas installées
# - RAG + FAISS + logs [PERF] (jury-ready)
# - Fallback embeddings: OpenAI -> (sinon) HuggingFace si disponible
# - FIX majeur : init OpenAI/ChatOpenAI/OpenAIEmbeddings "safe" (httpx client)
# =============================================================================


# ───────────────────────────── Optional imports ─────────────────────────────
def _try_import_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass


def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except Exception:
        return None


def _try_import_pypdf_loader():
    """
    PyPDFLoader est dans langchain_community. Si pas dispo, on désactive PDF.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader  # type: ignore

        return PyPDFLoader
    except Exception:
        return None


def _try_import_hf_embeddings():
    """
    Fallback embeddings HuggingFace si OpenAI absent.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore

        return HuggingFaceEmbeddings
    except Exception:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

            return HuggingFaceEmbeddings
        except Exception:
            return None


# Core LangChain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vectorstore FAISS
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
except Exception as e:
    raise RuntimeError(
        "Dépendance manquante: langchain_community (pour FAISS).\n"
        "Installe: pip install langchain-community faiss-cpu"
    ) from e

# OpenAI (optionnel)
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
except Exception:
    ChatOpenAI = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore


# ───────────────────────────── Env / Paths ─────────────────────────────
_try_import_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "Corpus"))).expanduser().resolve()

PDF_NAME = os.getenv("PDF_NAME", "Cours_Fractions_5e.pdf")
PDF_PATH = Path(os.getenv("PDF_PATH", str(DOCS_DIR / PDF_NAME))).expanduser().resolve()

ERREURS_XLSX = Path(os.getenv("ERREURS_XLSX", str(DOCS_DIR / "Erreurs_Fractions_5e.xlsx"))).expanduser().resolve()
REMED_XLSX = Path(os.getenv("REMED_XLSX", str(DOCS_DIR / "Remediations_Fractions_5e.xlsx"))).expanduser().resolve()

FAISS_DIR = Path(os.getenv("FAISS_DIR", str(BASE_DIR / "faiss_store"))).expanduser().resolve()
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")
FINGERPRINT_PATH = FAISS_DIR / "fingerprint.txt"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# ───────────────────────────── Logging / Perf ──────────────────────────
PERF_LOG = os.getenv("PERF_LOG", "1").strip().lower() not in {"0", "false", "no"}
PERF_LOG_JSON = os.getenv("PERF_LOG_JSON", "0").strip().lower() in {"1", "true", "yes"}

APP_VERSION = os.getenv("APP_VERSION", "").strip()
GIT_SHA = os.getenv("GIT_SHA", "").strip()


def _log(event: str, payload: Dict[str, Any]) -> None:
    if not PERF_LOG:
        return
    base = {"event": event, "ts": round(time.time(), 3)}
    if APP_VERSION:
        base["app_version"] = APP_VERSION
    if GIT_SHA:
        base["git_sha"] = GIT_SHA

    data = {**base, **payload}

    if PERF_LOG_JSON:
        print("[PERF] " + json.dumps(data, ensure_ascii=False), flush=True)
    else:
        parts = [f"{k}={data[k]}" for k in sorted(data.keys())]
        print("[PERF] " + " ".join(parts), flush=True)


# ───────────────────────────── Utils ───────────────────────────────────
def _ensure_dir_writable(dir_path: Path) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    test_file = dir_path / ".write_test"
    test_file.write_text("ok", encoding="utf-8")
    try:
        test_file.unlink()
    except Exception:
        pass


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
        try:
            st = p.stat()
            h.update(p.name.encode("utf-8"))
            h.update(str(st.st_mtime_ns).encode("utf-8"))
            h.update(str(st.st_size).encode("utf-8"))
        except Exception:
            continue

    return h.hexdigest()[:16]


def _corpus_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "docs_dir_exists": DOCS_DIR.exists(),
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


# ───────────────────────────── OpenAI SAFE CLIENT ─────────────────────────
# Le bug que tu vois vient souvent de versions openai/httpx où "proxies" a changé.
# On contourne en construisant un client OpenAI explicite avec httpx.Client.
_OPENAI_CLIENT = None
_OPENAI_HTTPX = None
_OPENAI_CLIENT_ERR: Optional[str] = None


def _get_openai_client():
    """
    Retourne un client OpenAI "safe" (OpenAI(api_key=..., http_client=httpx.Client(...))).
    Si openai n'est pas dispo ou si ça échoue -> None, avec _OPENAI_CLIENT_ERR rempli.
    """
    global _OPENAI_CLIENT, _OPENAI_HTTPX, _OPENAI_CLIENT_ERR

    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if _OPENAI_CLIENT_ERR is not None:
        return None

    if not OPENAI_API_KEY:
        _OPENAI_CLIENT_ERR = "OPENAI_API_KEY manquant."
        return None

    try:
        import httpx
        from openai import OpenAI

        _OPENAI_HTTPX = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
        )
        _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY, http_client=_OPENAI_HTTPX)
        return _OPENAI_CLIENT

    except Exception as e:
        _OPENAI_CLIENT_ERR = f"OpenAI client init failed: {type(e).__name__}: {e}"
        return None


def _inject_client_kwargs(cls, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Injecte (si possible) le client OpenAI et/ou le http_client dans les kwargs,
    selon la signature de la classe LangChain (qui varie selon versions).
    """
    kwargs = dict(base_kwargs)

    client = _get_openai_client()
    if client is None:
        return kwargs

    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        # Selon versions: "client" / "openai_client" / "http_client"
        if "client" in params:
            kwargs["client"] = client
        elif "openai_client" in params:
            kwargs["openai_client"] = client

        if _OPENAI_HTTPX is not None and "http_client" in params:
            kwargs["http_client"] = _OPENAI_HTTPX

    except Exception:
        # Si inspect échoue -> on ne force rien
        pass

    return kwargs


# ───────────────────────────── Embeddings / LLM ─────────────────────────
def _embeddings():
    """
    1) OpenAIEmbeddings si OPENAI_API_KEY est présent et langchain_openai installé
       -> avec client OpenAI "safe"
    2) sinon HuggingFaceEmbeddings si dispo
    3) sinon erreur claire
    """
    openai_model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")

    if OPENAI_API_KEY and OpenAIEmbeddings is not None:
        base_kwargs = {"model": openai_model, "api_key": OPENAI_API_KEY}
        kwargs = _inject_client_kwargs(OpenAIEmbeddings, base_kwargs)
        try:
            return OpenAIEmbeddings(**kwargs)
        except Exception as e:
            # Fallback HF si possible
            _log(
                "openai_embeddings_init_failed",
                {"error": f"{type(e).__name__}: {e}"},
            )

    HFEmb = _try_import_hf_embeddings()
    if HFEmb is not None:
        hf_model = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HFEmb(model_name=hf_model)

    raise RuntimeError(
        "Impossible d'initialiser les embeddings.\n"
        "- Option A (OpenAI): installe langchain-openai et définis OPENAI_API_KEY\n"
        "- Option B (HF local): pip install langchain-huggingface sentence-transformers\n"
        "\nSi tu as l'erreur 'proxies': fixe côté deps: pip install \"httpx==0.27.2\" --force-reinstall"
    )


def _llm():
    """
    LLM OpenAI obligatoire pour la génération ici.
    -> FIX 'proxies' : on injecte un client OpenAI 'safe' si possible.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY manquant. Ajoute-le dans ton .env ou dans les Secrets (HF).")
    if ChatOpenAI is None:
        raise RuntimeError(
            "Dépendance manquante: langchain-openai.\n"
            "Installe: pip install langchain-openai openai"
        )

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    base_kwargs = {"model": model, "temperature": temperature, "api_key": OPENAI_API_KEY}
    kwargs = _inject_client_kwargs(ChatOpenAI, base_kwargs)

    try:
        return ChatOpenAI(**kwargs)
    except Exception as e:
        # Message ultra clair + actionnable
        detail = f"{type(e).__name__}: {e}"
        client_err = _OPENAI_CLIENT_ERR
        raise RuntimeError(
            "Échec d'initialisation du LLM (ChatOpenAI).\n"
            f"- Erreur: {detail}\n"
            f"- OpenAI client err: {client_err or '(none)'}\n"
            "\n✅ Fix recommandé (Windows):\n"
            "1) pip install \"httpx==0.27.2\" --force-reinstall\n"
            "2) (option) pip install -U openai langchain-openai\n"
        ) from e


# ───────────────────────────── Load corpus ─────────────────────────────
def _load_txt_docs(docs_dir: Path) -> List[Document]:
    if not docs_dir.exists():
        return []
    docs: List[Document] = []
    for p in sorted(docs_dir.glob("*.txt")):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            continue
        if not txt:
            continue
        docs.append(Document(page_content=txt, metadata={"type": "txt", "source": str(p)}))
    return docs


def _load_pdf_docs(pdf_path: Path) -> List[Document]:
    PyPDFLoader = _try_import_pypdf_loader()
    if PyPDFLoader is None:
        _log("pdf_loader_missing", {"pdf_path": str(pdf_path)})
        return []
    if not pdf_path.exists():
        return []
    try:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for d in docs:
            md = d.metadata or {}
            md["type"] = "pdf"
            md["source"] = str(pdf_path)
            d.metadata = md
        return docs
    except Exception as e:
        _log("pdf_load_failed", {"pdf_path": str(pdf_path), "error": f"{type(e).__name__}: {e}"})
        return []


def _load_excel_as_docs(xlsx_path: Path, default_type: str) -> List[Document]:
    pd = _try_import_pandas()
    if pd is None:
        _log("pandas_missing_skip_xlsx", {"xlsx_path": str(xlsx_path)})
        return []
    if not xlsx_path.exists():
        return []
    try:
        xls = pd.ExcelFile(xlsx_path)
        sheet = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        _log("xlsx_load_failed", {"xlsx_path": str(xlsx_path), "error": f"{type(e).__name__}: {e}"})
        return []

    docs: List[Document] = []
    for idx, row in df.iterrows():
        parts: List[str] = []
        for col in df.columns:
            val = row[col]
            try:
                if pd.isna(val) or str(val).strip() == "":
                    continue
            except Exception:
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
            "❌ Aucun document trouvé pour le corpus.\n"
            f"- Mets au moins un .txt dans: {DOCS_DIR}\n"
            f"- PDF optionnel: {PDF_PATH}\n"
            f"- XLSX optionnels: {ERREURS_XLSX} / {REMED_XLSX}\n"
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
    try:
        FINGERPRINT_PATH.write_text(fp, encoding="utf-8")
    except Exception:
        pass

    t4 = time.perf_counter()

    _log(
        "vectorstore_build",
        {
            "corpus_fp": fp,
            "n_docs_raw": len(docs),
            "n_chunks": len(chunks),
            "chunk_size": int(os.getenv("CHUNK_SIZE", "650")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "110")),
            "embed_backend": "openai" if (OPENAI_API_KEY and OpenAIEmbeddings is not None) else "huggingface",
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
        try:
            fp_old = FINGERPRINT_PATH.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            fp_old = None

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
                {"corpus_fp": fp_now, "faiss_dir": str(FAISS_DIR), "t_load_s": round(t1 - t0, 3), **_corpus_stats()},
            )
            return _VECTORSTORE
        except Exception as e:
            _log(
                "vectorstore_load_failed",
                {"corpus_fp": fp_now, "faiss_dir": str(FAISS_DIR), "error": f"{type(e).__name__}: {e}"},
            )

    _VECTORSTORE = _build_vectorstore()
    return _VECTORSTORE


# ───────────────────────────── Prompt / Context ─────────────────────────
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


def _rank_docs(docs: List[Document]) -> List[Document]:
    order = {"txt": 0, "pdf": 1, "excel": 2}
    return sorted(docs, key=lambda d: order.get((d.metadata or {}).get("type", "unknown"), 9))


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


def _retriever_params() -> Dict[str, Any]:
    return {
        "search_type": "mmr",
        "k": int(os.getenv("RETRIEVER_K", "5")),
        "fetch_k": int(os.getenv("RETRIEVER_FETCH_K", "25")),
        "lambda_mult": float(os.getenv("RETRIEVER_LAMBDA", "0.7")),
    }


def _context_stats(docs: List[Document], context: str) -> Dict[str, Any]:
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


# ───────────────────────────── Public API ──────────────────────────────
def rag_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    rag_chain({"question": "..."}) -> {"answer": str, "source_documents": List[Document]}
    """
    question = (inputs.get("question") or "").strip()
    if not question:
        return {"answer": "Je ne sais pas.", "source_documents": []}

    t0 = time.perf_counter()

    store = _load_or_create_vectorstore()
    t1 = time.perf_counter()

    rp = _retriever_params()
    retriever = store.as_retriever(
        search_type=rp["search_type"],
        search_kwargs={"k": rp["k"], "fetch_k": rp["fetch_k"], "lambda_mult": rp["lambda_mult"]},
    )

    source_docs = retriever.invoke(question)
    t2 = time.perf_counter()

    source_docs = _rank_docs(source_docs)
    context = _format_context(source_docs)

    llm = _llm()
    messages = _RAG_PROMPT.format_messages(question=question, context=context)

    msg = llm.invoke(messages)
    t3 = time.perf_counter()

    answer = (getattr(msg, "content", None) or "").strip() or "Je ne sais pas."

    q_hash = hashlib.sha1(question.encode("utf-8", errors="ignore")).hexdigest()[:10]
    cfg = {
        "corpus_fp": _corpus_fingerprint(),
        "embed_backend": "openai" if (OPENAI_API_KEY and OpenAIEmbeddings is not None) else "huggingface",
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

    _log("rag_config", {"q_hash": q_hash, "q_len": len(question), **cfg, **ctx_stats})
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