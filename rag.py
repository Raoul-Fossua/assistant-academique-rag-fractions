#!/usr/bin/env python3
"""
rag.py
────────────────────────────
RAG minimal (FAISS + OpenAI) en standalone (script console).

⚠️ Ton app Chainlit utilise plutôt rag_langchain.py + agents.py.
Ce fichier est conservé propre au cas où tu veux tester en local.

✅ Ingestion PDF + TXT
✅ Chunking
✅ FAISS persistant
✅ Citations
"""

from __future__ import annotations

import os
import json
import textwrap
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tqdm.auto import tqdm

load_dotenv()

# ─────────────────────────── Config ──────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Corpus (corrigé: Corpus avec C majuscule)
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "Corpus")))

# Vectorstore persistant (corrigé: nom cohérent)
VSTORE_DIR = Path(os.getenv("VSTORE_DIR", str(BASE_DIR / "vectorstore")))
VSTORE_DIR.mkdir(parents=True, exist_ok=True)

FAISS_PATH = VSTORE_DIR / "faiss.index"
META_PATH = VSTORE_DIR / "faiss.meta.json"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
TOP_K = int(os.getenv("TOP_K", "8"))

SYSTEM_PROMPT = (
    "Tu es un assistant pédagogique de mathématiques, précis et utile.\n"
    "Règles:\n"
    "1) Réponds UNIQUEMENT à partir du contexte fourni.\n"
    "2) Si l’information n’est pas dans le contexte, dis clairement: « Je ne sais pas. »\n"
    "3) Donne une réponse structurée: explication courte + exemple si pertinent.\n"
    "4) Termine par des citations de sources sous la forme: [source:page].\n"
    "5) Ne fabrique jamais de références.\n"
)

openai.api_key = os.getenv("OPENAI_API_KEY", "").strip()


# ─────────────────────────── Data model ──────────────────────────
@dataclass
class Chunk:
    text: str
    meta: Dict[str, Any]  # {"source": "...", "page": 3, "type": "pdf/txt", ...}


def _stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


# ─────────────────────────── Ingestion ───────────────────────────
def read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_index_1based, text)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append((i, page.extract_text() or ""))
    return pages


def load_and_split() -> List[Chunk]:
    if not DOCS_DIR.exists():
        raise RuntimeError(f"DOCS_DIR introuvable: {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    chunks: List[Chunk] = []
    for path in DOCS_DIR.rglob("*"):
        suf = path.suffix.lower()

        if suf == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
            for part in splitter.split_text(text):
                chunks.append(
                    Chunk(
                        text=part,
                        meta={
                            "source": str(path.relative_to(DOCS_DIR)),
                            "page": None,
                            "type": "txt",
                        },
                    )
                )

        elif suf == ".pdf":
            for page_no, page_text in read_pdf_pages(path):
                if not page_text.strip():
                    continue
                for part in splitter.split_text(page_text):
                    chunks.append(
                        Chunk(
                            text=part,
                            meta={
                                "source": str(path.relative_to(DOCS_DIR)),
                                "page": page_no,
                                "type": "pdf",
                            },
                        )
                    )

    if not chunks:
        raise RuntimeError(f"Aucun .txt ou .pdf trouvé dans {DOCS_DIR}")
    return chunks


# ───────────────────────── Embeddings ────────────────────────────
def embed(texts: List[str]) -> List[List[float]]:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY manquant (env).")
    res = openai.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in res.data]


# ───────────────────────── Vector store ──────────────────────────
def load_store() -> Tuple[faiss.Index, List[Chunk]]:
    index = faiss.read_index(str(FAISS_PATH))
    raw = json.loads(META_PATH.read_text(encoding="utf-8"))
    chunks = [Chunk(text=c["text"], meta=c["meta"]) for c in raw]
    return index, chunks


def save_store(index: faiss.Index, chunks: List[Chunk]) -> None:
    faiss.write_index(index, str(FAISS_PATH))
    META_PATH.write_text(
        json.dumps([{"text": c.text, "meta": c.meta} for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )


def get_faiss_store(chunks: List[Chunk]) -> Tuple[faiss.Index, List[Chunk]]:
    if FAISS_PATH.exists() and META_PATH.exists():
        print("✓ Loading existing vector store …")
        return load_store()

    print("⏳ Building vector store …")
    all_vectors: List[List[float]] = []
    texts = [c.text for c in chunks]

    for i in tqdm(range(0, len(texts), 128), unit="batch"):
        all_vectors.extend(embed(texts[i : i + 128]))

    mat = np.asarray(all_vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(mat.shape[1])
    index.add(mat)

    save_store(index, chunks)
    return index, chunks


# ───────────────────────── Retrieval ─────────────────────────────
def retrieve(query: str, index: faiss.Index, chunks: List[Chunk], k: int = TOP_K) -> List[Chunk]:
    q_vec = np.asarray(embed([query])[0], dtype=np.float32).reshape(1, -1)
    _, idxs = index.search(q_vec, k)
    out: List[Chunk] = []
    for i in idxs[0]:
        if i < 0:
            continue
        out.append(chunks[int(i)])
    return out


def format_sources(ctx: List[Chunk]) -> str:
    seen = set()
    refs = []
    for c in ctx:
        src = c.meta.get("source", "unknown")
        page = c.meta.get("page", None)
        ref = f"{src}:{page}" if page is not None else f"{src}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return " ; ".join(f"[{r}]" for r in refs[:8])


def build_user_prompt(question: str, ctx_chunks: List[Chunk]) -> str:
    context_block = "\n\n".join(
        f"[Doc {i+1} | {c.meta.get('source')} | page {c.meta.get('page')}]\n{c.text}"
        for i, c in enumerate(ctx_chunks)
    )
    return (
        "Contexte (extraits de documents pédagogiques):\n"
        f"{context_block}\n\n"
        f"Question: {question}\n\n"
        "Réponds en t'appuyant uniquement sur le contexte. "
        "Si tu ne peux pas répondre, dis « Je ne sais pas. »"
    )


def chat_loop(index: faiss.Index, chunks: List[Chunk]) -> None:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY manquant (env).")

    history: List[Dict[str, str]] = []
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    while True:
        try:
            q = input("\n💬 Question (Ctrl-C pour quitter): ").strip()
        except KeyboardInterrupt:
            print("\nBye!")
            break

        if not q:
            continue

        ctx = retrieve(q, index, chunks)
        user_prompt = build_user_prompt(q, ctx)

        messages = [system_msg] + history + [{"role": "user", "content": user_prompt}]
        response = openai.chat.completions.create(
            model=CHAT_MODEL, messages=messages, temperature=0.2
        )
        answer = response.choices[0].message.content.strip()

        srcs = format_sources(ctx)
        if answer and ("Sources:" not in answer):
            answer = f"{answer}\n\nSources: {srcs}"

        print("🤖 Réponse:\n")
        print(textwrap.fill(answer, width=88))

        history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": answer}])


if __name__ == "__main__":
    chunks = load_and_split()
    index, chunks = get_faiss_store(chunks)
    chat_loop(index, chunks)
