from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ───────────────────────────── Config ─────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise SystemExit("❌ OPENAI_API_KEY manquant. Mets-le dans .env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "corpus")))
PDF_NAME = os.getenv("PDF_NAME", "Cours_Fractions_5e.pdf")
PDF_PATH = DOCS_DIR / PDF_NAME

ERREURS_XLSX = Path(os.getenv("ERREURS_XLSX", str(DOCS_DIR / "Erreurs_Fractions_5e.xlsx")))
REMED_XLSX = Path(os.getenv("REMED_XLSX", str(DOCS_DIR / "Remediations_Fractions_5e.xlsx")))

RESPONSES_CSV = Path(os.getenv("RESPONSES_CSV", str(BASE_DIR / "data" / "students" / "responses.csv")))

# ───────────────────────────── LLM ────────────────────────────────
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    api_key=OPENAI_API_KEY,
)

# ───────────────────────────── RAG ────────────────────────────────
# rag_langchain.py doit exposer `rag_chain(payload: dict) -> dict`
# qui renvoie au minimum:
#   {"answer": str, "source_documents": List[Document]}
from rag_langchain import rag_chain

# ───────────────────────────── Tools ──────────────────────────────
# On utilise le décorateur @tool (stable dans LangChain).
from langchain.tools import tool


def _fmt_source(doc) -> str:
    """Formate une source (pdf/excel/...) de façon lisible."""
    meta = getattr(doc, "metadata", None) or {}
    src = meta.get("source") or meta.get("file_name") or meta.get("basename") or "unknown"
    src_name = os.path.basename(str(src))

    doc_type = meta.get("type")

    # PDF
    if doc_type == "pdf" and meta.get("page") is not None:
        try:
            page = int(meta["page"]) + 1
        except Exception:
            page = meta["page"]
        return f"{src_name}:{page}"

    # Excel
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


@tool
def fractions_rag(question: str) -> str:
    """
    Répond sur les FRACTIONS (niveau 5e) uniquement à partir du corpus local (PDF + Excel).
    Retourne toujours des sources.
    """
    # petit check utile (évite les réponses "vides" si l'utilisateur n'a pas mis les fichiers)
    if not DOCS_DIR.exists():
        return f"❌ Dossier corpus introuvable: {DOCS_DIR}"
    if not PDF_PATH.exists():
        return f"❌ PDF introuvable: {PDF_PATH}"

    result = rag_chain({"question": question})
    answer = (result.get("answer") or "").strip() or "Je ne sais pas."
    sources = result.get("source_documents") or []
    return f"{answer}\n\n{_sources_block(sources)}"


@tool
def didactic_check(text: str) -> str:
    """
    Réécrit un contenu en version didactique (fractions 5e) : sens, exemple, erreur fréquente.
    """
    prompt = f"""
Tu es un didacticien en mathématiques (spécialiste des fractions, niveau 5e).
Améliore le texte en évitant les "règles magiques".

Structure obligatoire :
1) Idée clé
2) Explication (avec sens)
3) Mini-exemple
4) Erreur fréquente + comment l’éviter

Texte :
{text}

Réécriture :
"""
    return llm.invoke(prompt).content.strip()


def _load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    xls = pd.ExcelFile(path)
    sheet = xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet)


@tool
def lookup_error_remediation(error_id: str) -> str:
    """
    Récupère une erreur + remédiation via error_id dans les Excel (erreurs/remédiations).
    """
    eid = (error_id or "").strip()
    if not eid:
        return "❌ Donne un error_id (ex: add_denominators)."

    err_df = _load_excel(ERREURS_XLSX)
    rem_df = _load_excel(REMED_XLSX)

    if err_df.empty or rem_df.empty:
        return (
            "❌ Excel introuvables ou vides.\n"
            f"- {ERREURS_XLSX}\n"
            f"- {REMED_XLSX}"
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


@tool
def groups_from_csv(_: str = "") -> str:
    """
    Pré-analyse simple : agrège les error_tags par élève (responses.csv),
    puis affiche les 3 tags dominants.
    """
    if not RESPONSES_CSV.exists():
        return (
            f"❌ Fichier introuvable: {RESPONSES_CSV}\n"
            "Crée `data/students/responses.csv` avec au minimum: student_id, error_tags."
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
    agg["top_tags"] = agg["tags_list"].apply(
        lambda L: ", ".join(pd.Series(L).value_counts().head(3).index.tolist())
    )

    lines = ["🧩 **Pré-analyse des profils d’erreurs (Fractions 5e)**\n"]
    for _, row in agg.iterrows():
        lines.append(f"- {row['student_id']} → {row['top_tags'] or '(aucune erreur taguée)'}")

    lines.append(
        "\n👉 Pour un clustering complet, utilise `clustering_fractions.ipynb` "
        "puis exporte un `groups_of_need.csv`."
    )
    return "\n".join(lines)


@tool
def web_search_tavily(query: str) -> str:
    """
    Recherche web via Tavily (utile pour enrichir / vérifier des points).
    Si TAVILY_API_KEY est absent, l'outil explique quoi faire.
    """
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


TOOLS = [
    fractions_rag,
    didactic_check,
    lookup_error_remediation,
    groups_from_csv,
    web_search_tavily,
]

# ───────────────────────────── Agent ──────────────────────────────
# IMPORTANT : on utilise create_agent (que tu as déjà dans ton env).
# Zéro import AgentExecutor => fini le crash.
from langchain.agents import create_agent


SYSTEM_PROMPT = f"""
Tu es un assistant pédagogique intelligent spécialisé sur les FRACTIONS (niveau 5e),
dans le cadre d’un mémoire DU Sorbonne Data Analytics.

Corpus local attendu :
- PDF cours: {PDF_PATH}
- Excel erreurs: {ERREURS_XLSX}
- Excel remédiations: {REMED_XLSX}

Règles de décision :
- Pour une question de cours/méthode/erreur/remédiation : utilise d’abord l’outil `fractions_rag`.
- Pour rendre une explication plus pédagogique : utilise `didactic_check`.
- Pour récupérer une fiche via un `error_id` : utilise `lookup_error_remediation`.
- Pour analyse classe : utilise `groups_from_csv`.
- Pour une vérification web (hors corpus) : utilise `web_search_tavily` (si disponible).

Règle de vérité :
- Si l’info n’est pas dans le corpus et que la recherche web n’est pas dispo : dis exactement « Je ne sais pas. »
- N’invente jamais de sources.
""".strip()

# create_agent retourne un runnable “agent” (pas besoin d’AgentExecutor)
agent = create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def _extract_text(result: Any) -> str:
    """Récupère proprement du texte depuis différents formats de retour."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        # selon versions, la clé peut varier
        for k in ("output", "final", "answer", "result", "text"):
            v = result.get(k)
            if isinstance(v, str) and v.strip():
                return v
        # fallback : stringify
        return str(result)
    # messages / objets
    return str(result)


def run_agent(message: str) -> str:
    """
    Fonction appelée par Chainlit.
    Retourne toujours une réponse texte.
    """
    msg = (message or "").strip()
    if not msg:
        return "Écris une question 🙂"

    # IMPORTANT : certaines implémentations attendent {"input": "..."}
    # d’autres acceptent directement une string.
    try:
        if hasattr(agent, "invoke"):
            try:
                res = agent.invoke({"input": msg})
            except Exception:
                res = agent.invoke(msg)
        else:
            # fallback ultra-safe
            return fractions_rag(msg)
    except Exception as e:
        # On ne laisse pas l’UI mourir
        return (
            "⚠️ Désolé, je n’ai pas pu générer de réponse.\n\n"
            f"Erreur: {type(e).__name__}: {e}"
        )

    out = _extract_text(res).strip()
    return out if out else "Désolé, je n’ai pas pu générer de réponse."
