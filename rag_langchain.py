# rag_langchain.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from dotenv import load_dotenv

# ───────────────────────────── Env / Paths ─────────────────────────────
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Corpus
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "corpus")))

PDF_PATH = Path(os.getenv("PDF_PATH", str(DOCS_DIR / "Cours_Fractions_5e.pdf")))
ERREURS_XLSX = Path(os.getenv("ERREURS_XLSX", str(DOCS_DIR / "Erreurs_Fractions_5e.xlsx")))
REMED_XLSX = Path(os.getenv("REMED_XLSX", str(DOCS_DIR / "Remediations_Fractions_5e.xlsx")))

# IMPORTANT: FAISS_DIR configurable pour éviter les chemins trop longs sous Windows
FAISS_DIR = Path(os.getenv("FAISS_DIR", str(BASE_DIR / "faiss_store"))).expanduser().resolve()
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

# API key (obligatoire pour embeddings + LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise SystemExit("❌ OPENAI_API_KEY manquant. Mets-le dans .env (OPENAI_API_KEY=sk-...).")

# ───────────────────────────── LangChain imports ───────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# ───────────────────────────── Utils ───────────────────────────────────
def _ensure_dir_writable(dir_path: Path) -> None:
    """
    Vérifie qu'on peut écrire dans un dossier (Windows-friendly).
    Lève une erreur explicite si non.
    """
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        test_file = dir_path / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)
    except Exception as e:
        raise RuntimeError(
            "❌ Impossible d’écrire dans FAISS_DIR.\n"
            f"FAISS_DIR = {dir_path}\n"
            "➡️ Mets un chemin court et accessible dans .env, ex:\n"
            "   FAISS_DIR=C:\\faiss_fractions\n"
            f"Détail: {e}"
        )


def _embeddings() -> OpenAIEmbeddings:
    """
    Embeddings OpenAI (modèle moderne).
    """
    model = os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model, api_key=OPENAI_API_KEY)


def _llm() -> ChatOpenAI:
    """
    LLM OpenAI pour la synthèse de réponse.
    """
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0"))
    return ChatOpenAI(model=model, temperature=temperature, api_key=OPENAI_API_KEY)


# ───────────────────────────── Load corpus ─────────────────────────────
def _load_pdf_docs(pdf_path: Path) -> List[Document]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"❌ PDF introuvable: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()  # Document par page
    for d in docs:
        md = d.metadata or {}
        md["type"] = "pdf"
        md["source"] = str(pdf_path)
        d.metadata = md
    return docs


def _load_excel_as_docs(xlsx_path: Path, default_type: str) -> List[Document]:
    """
    Convertit chaque ligne Excel en Document.
    On prend la première feuille par défaut.
    """
    if not xlsx_path.exists():
        return []

    xls = pd.ExcelFile(xlsx_path)
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)

    docs: List[Document] = []
    for idx, row in df.iterrows():
        parts = []
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
                    "subtype": default_type,  # erreurs / remediations
                    "source": str(xlsx_path),
                    "sheet": sheet,
                    "row": int(idx) + 2,  # header + 1-based
                },
            )
        )
    return docs


def _load_corpus_docs() -> List[Document]:
    pdf_docs = _load_pdf_docs(PDF_PATH)
    err_docs = _load_excel_as_docs(ERREURS_XLSX, default_type="erreurs")
    rem_docs = _load_excel_as_docs(REMED_XLSX, default_type="remediations")
    return pdf_docs + err_docs + rem_docs


# ───────────────────────────── Chunking ────────────────────────────────
def _split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
    )
    return splitter.split_documents(docs)


# ───────────────────────────── Vectorstore ─────────────────────────────
_VECTORSTORE: Optional[FAISS] = None


def _build_vectorstore() -> FAISS:
    docs = _load_corpus_docs()
    chunks = _split_docs(docs)

    store = FAISS.from_documents(chunks, _embeddings())

    # ✅ CRUCIAL: créer le dossier avant save_local
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_dir_writable(FAISS_DIR)

    store.save_local(str(FAISS_DIR), index_name=FAISS_INDEX_NAME)
    return store


def _load_or_create_vectorstore() -> FAISS:
    global _VECTORSTORE
    if _VECTORSTORE is not None:
        return _VECTORSTORE

    if FAISS_DIR.exists():
        try:
            _VECTORSTORE = FAISS.load_local(
                str(FAISS_DIR),
                _embeddings(),
                index_name=FAISS_INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            return _VECTORSTORE
        except Exception:
            pass  # on reconstruit

    _VECTORSTORE = _build_vectorstore()
    return _VECTORSTORE


# ───────────────────────────── RAG pipeline ────────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Tu es un assistant pédagogique spécialisé en FRACTIONS niveau 5e.\n"
                "Tu dois répondre UNIQUEMENT à partir du CONTEXTE fourni.\n"
                "Si le contexte ne suffit pas, dis exactement: « Je ne sais pas. »\n"
                "Réponse claire, structurée, courte.\n"
            ),
        ),
        ("human", "Question: {question}\n\nCONTEXTE:\n{context}"),
    ]
)


def _format_context(docs: List[Document]) -> str:
    blocks = []
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
        else:
            header = f"[{t} {src}]"

        blocks.append(header + "\n" + d.page_content)

    return "\n\n---\n\n".join(blocks)


def rag_chain(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interface:
    rag_chain({"question": "..."}) -> {"answer": str, "source_documents": List[Document]}
    """
    question = (inputs.get("question") or "").strip()
    if not question:
        return {"answer": "Je ne sais pas.", "source_documents": []}

    store = _load_or_create_vectorstore()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": int(os.getenv("RETRIEVER_K", "5"))},
    )

    # ✅ LangChain récent: on utilise invoke()
    source_docs = retriever.invoke(question)

    context = _format_context(source_docs)

    llm = _llm()
    prompt = _RAG_PROMPT.format_messages(question=question, context=context)
    msg = llm.invoke(prompt)
    answer = (getattr(msg, "content", None) or "").strip()

    if not answer:
        answer = "Je ne sais pas."

    return {"answer": answer, "source_documents": source_docs}
