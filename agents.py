from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────── Config ─────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise SystemExit("❌ OPENAI_API_KEY manquant. Mets-le dans .env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

BASE_DIR = Path(__file__).resolve().parent

# On supporte plusieurs cas : data/Corpus, data/corpus, data/CORPUS...
DEFAULT_DOCS_DIR = BASE_DIR / "data" / "Corpus"
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(DEFAULT_DOCS_DIR)))

PDF_NAME = os.getenv("PDF_NAME", "Cours_Fractions_5e.pdf")
PDF_PATH = DOCS_DIR / PDF_NAME

ERREURS_XLSX = Path(os.getenv("ERREURS_XLSX", str(DOCS_DIR / "Erreurs_Fractions_5e.xlsx")))
REMED_XLSX = Path(os.getenv("REMED_XLSX", str(DOCS_DIR / "Remediations_Fractions_5e.xlsx")))

# responses.csv (ton arbo actuelle : data/Students/)
DEFAULT_RESPONSES = BASE_DIR / "data" / "Students" / "responses.csv"
RESPONSES_CSV = Path(os.getenv("RESPONSES_CSV", str(DEFAULT_RESPONSES)))

# ───────────────────────────── Helpers paths ──────────────────────
def _resolve_existing_dir(candidate: Path) -> Path:
    """Si le dossier n'existe pas, tente de retrouver une variante de casse sur Windows."""
    if candidate.exists():
        return candidate

    # Essaye variantes courantes
    variants = [
        BASE_DIR / "data" / "Corpus",
        BASE_DIR / "data" / "corpus",
        BASE_DIR / "data" / "CORPUS",
    ]
    for v in variants:
        if v.exists():
            return v

    return candidate  # on renvoie quand même (le message d'erreur sera clair)

DOCS_DIR = _resolve_existing_dir(DOCS_DIR)
PDF_PATH = DOCS_DIR / PDF_NAME

# ───────────────────────────── LLM ────────────────────────────────
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    api_key=OPENAI_API_KEY,
)

# ───────────────────────────── RAG ────────────────────────────────
# rag_langchain.py doit exposer `rag_chain(payload: dict) -> dict`
# -> {"answer": str, "source_documents": List[Document]}
from rag_langchain import rag_chain

# ───────────────────────────── Utils sources ──────────────────────
def _fmt_source(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    src = meta.get("source") or meta.get("file_name") or meta.get("basename") or "unknown"
    src_name = os.path.basename(str(src))

    doc_type = meta.get("type")

    if doc_type == "pdf" and meta.get("page") is not None:
        try:
            page = int(meta["page"]) + 1
        except Exception:
            page = meta["page"]
        return f"{src_name}:{page}"

    if doc_type == "excel":
        sheet = meta.get("sheet", "sheet?")
        row = meta.get("row", None)
        if row is not None:
            return f"{src_name}|{sheet}|row={row}"
        return f"{src_name}|{sheet}"

    return src_name


def _sources_block(source_documents: List[Any]) -> str:
    if not source_documents:
        return "Sources: (aucune)"
    seen, refs = set(), []
    for d in source_documents:
        r = _fmt_source(d)
        if r not in seen:
            seen.add(r)
            refs.append(r)
    refs = refs[:10]
    return "Sources: " + " ; ".join(f"[{r}]" for r in refs)


def _missing_corpus_message() -> str:
    return (
        "❌ **Corpus introuvable / incomplet**\n\n"
        f"- Dossier attendu : `{DOCS_DIR}`\n"
        f"- PDF attendu : `{PDF_PATH}`\n"
        f"- Excel erreurs : `{ERREURS_XLSX}`\n"
        f"- Excel remédiations : `{REMED_XLSX}`\n\n"
        "✅ Vérifie :\n"
        "1) que `data/Corpus/` existe bien\n"
        "2) que les fichiers PDF/XLSX sont dedans\n"
        "3) que tes variables `.env` (DOCS_DIR/PDF_NAME/ERREURS_XLSX/REMED_XLSX) correspondent.\n"
    )

# ───────────────────────────── Tools “maison” ─────────────────────
def fractions_rag(question: str) -> str:
    """Réponse Fractions 5e basée sur le corpus local + sources."""
    if not DOCS_DIR.exists() or not PDF_PATH.exists():
        return _missing_corpus_message()

    result = rag_chain({"question": question})
    answer = (result.get("answer") or "").strip() or "Je ne sais pas."
    sources = result.get("source_documents") or []
    return f"{answer}\n\n{_sources_block(sources)}"


def didactic_check(text: str) -> str:
    """Réécriture didactique (sens, exemple, erreur fréquente)."""
    prompt = f"""
Tu es un didacticien en mathématiques (spécialiste des fractions, niveau 5e).
Améliore le texte en évitant les "règles magiques" et en donnant du sens.

Structure obligatoire :
1) Idée clé
2) Explication (avec sens)
3) Mini-exemple
4) Erreur fréquente + comment l’éviter

Texte à transformer :
{text}

Réponse :
""".strip()
    return llm.invoke(prompt).content.strip()


def _load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet)


def lookup_error_remediation(error_id: str) -> str:
    eid = (error_id or "").strip()
    if not eid:
        return "❌ Donne un error_id (ex: add_denominators)."

    err_df = _load_excel(ERREURS_XLSX)
    rem_df = _load_excel(REMED_XLSX)

    if err_df.empty or rem_df.empty:
        return (
            "❌ Excel introuvables ou vides.\n"
            f"- {ERREURS_XLSX}\n"
            f"- {REMED_XLSX}\n"
            "\n👉 Remets les fichiers dans `data/Corpus/` ou corrige les chemins `.env`."
        )

    err_df.columns = [c.strip().lower() for c in err_df.columns]
    rem_df.columns = [c.strip().lower() for c in rem_df.columns]

    if "error_id" not in err_df.columns or "error_id" not in rem_df.columns:
        return "❌ Les fichiers Excel doivent contenir une colonne `error_id`."

    err = err_df[err_df["error_id"].astype(str).str.strip() == eid]
    rem = rem_df[rem_df["error_id"].astype(str).str.strip() == eid]

    if err.empty and rem.empty:
        return f"Je ne trouve pas l’error_id: {eid}"

    out = [f"🔎 error_id = {eid}\n"]

    if not err.empty:
        r = err.iloc[0].to_dict()
        out.append("📌 **Erreur (Excel)**")
        for k, v in r.items():
            if pd.isna(v) or str(v).strip() == "":
                continue
            out.append(f"- {k}: {v}")
        out.append(f"Source: [{ERREURS_XLSX.name}]")

    if not rem.empty:
        r = rem.iloc[0].to_dict()
        out.append("\n🛠️ **Remédiation (Excel)**")
        for k, v in r.items():
            if pd.isna(v) or str(v).strip() == "":
                continue
            out.append(f"- {k}: {v}")
        out.append(f"Source: [{REMED_XLSX.name}]")

    return "\n".join(out)


def groups_from_csv() -> str:
    if not RESPONSES_CSV.exists():
        return (
            f"❌ Fichier introuvable: `{RESPONSES_CSV}`\n\n"
            "✅ Pour l’activer :\n"
            "- crée `data/Students/responses.csv`\n"
            "- colonnes minimales : `student_id, error_tags`\n"
            "- `error_tags` séparés par `|` (ex: add_denominators|compare_fractions)\n"
        )

    df = pd.read_csv(RESPONSES_CSV)
    required = {"student_id", "error_tags"}
    if not required.issubset(df.columns):
        return (
            "❌ Colonnes manquantes dans responses.csv.\n"
            "Attendu au minimum: student_id, error_tags\n"
            f"Colonnes trouvées: {list(df.columns)}"
        )

    def split_tags(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [t.strip() for t in str(x).split("|") if t.strip()]

    df["tags_list"] = df["error_tags"].apply(split_tags)
    agg = df.groupby("student_id")["tags_list"].sum().reset_index()

    def top3(tags: List[str]) -> str:
        if not tags:
            return "(aucune erreur taguée)"
        vc = pd.Series(tags).value_counts().head(3)
        return ", ".join(vc.index.tolist())

    agg["top_tags"] = agg["tags_list"].apply(top3)

    lines = ["🧩 **Pré-analyse des profils d’erreurs (Fractions 5e)**\n"]
    for _, row in agg.iterrows():
        lines.append(f"- {row['student_id']} → {row['top_tags']}")

    lines.append(
        "\n👉 Pour un clustering complet, utilise `clustering_fractions.ipynb` "
        "puis exporte un `groups_of_need.csv`."
    )
    return "\n".join(lines)


def web_search_tavily(query: str) -> str:
    if not TAVILY_API_KEY:
        return (
            "❌ TAVILY_API_KEY manquant dans .env.\n"
            "➡️ Ajoute TAVILY_API_KEY=... (ou ignore cet outil)."
        )

    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    res = client.search(
        query=query,
        search_depth="basic",
        max_results=5,
        include_answer=True,
        include_raw_content=False,
    )

    answer = (res.get("answer") or "").strip()
    results = res.get("results") or []

    lines = []
    if answer:
        lines.append(f"🧭 **Synthèse Tavily**\n{answer}\n")

    if results:
        lines.append("🔗 **Résultats**")
        for r in results[:5]:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            snippet = (r.get("content") or "").strip()
            if title and url:
                lines.append(f"- {title}\n  {url}\n  {snippet}")

    return "\n".join(lines).strip() or "Aucun résultat Tavily."


# ───────────────────────────── Router ─────────────────────────────
def _looks_like_groups_request(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["responses.csv", "profil", "profils", "groupe", "groupes", "besoin", "classe", "clustering"])


def _extract_error_id(text: str) -> Optional[str]:
    # Ex: "error_id=add_denominators" ou "error_id: add_denominators"
    m = re.search(r"error_id\s*[:=]\s*([a-z0-9_\-]+)", text.strip(), flags=re.I)
    return m.group(1) if m else None


def _looks_like_didactic_request(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in ["rends didactique", "version didactique", "reformule", "vulgarise", "explique avec du sens"])


def _strip_didactic_prefix(text: str) -> str:
    # enlève "Rends didactique : ..." pour ne garder que le contenu à transformer
    s = text.strip()
    s = re.sub(r"^(rends\s+didactique\s*[:\-]?\s*)", "", s, flags=re.I)
    s = re.sub(r"^(reformule\s*[:\-]?\s*)", "", s, flags=re.I)
    s = re.sub(r"^(version\s+didactique\s*[:\-]?\s*)", "", s, flags=re.I)
    return s.strip() or text.strip()


def _looks_like_web_request(text: str) -> bool:
    t = text.lower()
    return t.startswith("/web ") or any(k in t for k in ["cherche sur le web", "recherche web", "sur internet", "tavily"])


def _strip_web_prefix(text: str) -> str:
    t = text.strip()
    if t.lower().startswith("/web"):
        return t[4:].strip()
    return t


# ───────────────────────────── API Chainlit ───────────────────────
def run_agent(message: str) -> str:
    """
    Point d’entrée unique (Chainlit).
    Objectif: toujours renvoyer une STRING utile, et ne jamais crasher.
    """
    msg = (message or "").strip()
    if not msg:
        return "Écris une question 🙂"

    # 0) commandes rapides (optionnelles)
    if msg.strip().lower() in {"/groups", "/classe"}:
        return groups_from_csv()

    # 1) Groupes classe
    if _looks_like_groups_request(msg):
        return groups_from_csv()

    # 2) Lookup Excel (error_id)
    eid = _extract_error_id(msg)
    if eid:
        return lookup_error_remediation(eid)

    # 3) Didactique
    if _looks_like_didactic_request(msg):
        core = _strip_didactic_prefix(msg)
        return didactic_check(core)

    # 4) Web (optionnel)
    if _looks_like_web_request(msg):
        q = _strip_web_prefix(msg)
        if not q:
            return "❌ Utilise : `/web ta question`"
        return web_search_tavily(q)

    # 5) Par défaut: RAG local (cœur de ton assistant)
    try:
        return fractions_rag(msg)
    except Exception as e:
        return (
            "⚠️ Désolé, je n’ai pas pu générer de réponse via le RAG.\n\n"
            f"Erreur: {type(e).__name__}: {e}"
        )
