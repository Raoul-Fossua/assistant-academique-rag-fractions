from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# ───────────────────────────── Config ─────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

# ✅ Dossiers (corrigé: Corpus/Students en CamelCase)
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "Corpus"))).expanduser().resolve()
STUDENTS_DIR = Path(os.getenv("STUDENTS_DIR", str(BASE_DIR / "data" / "Students"))).expanduser().resolve()

PDF_NAME = os.getenv("PDF_NAME", "Cours_Fractions_5e.pdf")
PDF_PATH = (DOCS_DIR / PDF_NAME).resolve()

# Excel optionnels
ERREURS_XLSX = Path(os.getenv("ERREURS_XLSX", str(DOCS_DIR / "Erreurs_Fractions_5e.xlsx"))).expanduser().resolve()
REMED_XLSX = Path(os.getenv("REMED_XLSX", str(DOCS_DIR / "Remediations_Fractions_5e.xlsx"))).expanduser().resolve()

# Analyse classe
RESPONSES_CSV = Path(os.getenv("RESPONSES_CSV", str(STUDENTS_DIR / "responses.csv"))).expanduser().resolve()
SAMPLE_RESPONSES_CSV = Path(os.getenv("SAMPLE_RESPONSES_CSV", str(STUDENTS_DIR / "sample_responses.csv"))).expanduser().resolve()

EXPORTS_DIR = Path(os.getenv("EXPORTS_DIR", str(BASE_DIR / "exports"))).expanduser().resolve()

OBJ_COUNT = int(os.getenv("OBJ_COUNT", "10"))
TOP_HARD = int(os.getenv("TOP_HARD", "4"))

THRESH_A = int(os.getenv("THRESH_A", "9"))   # >=9/10
THRESH_B = int(os.getenv("THRESH_B", "7"))   # 7-8/10
THRESH_C = int(os.getenv("THRESH_C", "5"))   # 5-6/10
THRESH_D = int(os.getenv("THRESH_D", "3"))   # 3-4/10

PROFILE_MAP = {
    "Rep_Score": [1, 2],
    "Compare_Score": [3, 4],
    "Equiv_Score": [5, 6],
    "Ops_Score": [7, 8, 9, 10],
}

GROUP_INFO = {
    "A": ("Approfondissement (experts)", "Vert foncé", "#0B6E4F",
          "Défis : problèmes ouverts, justification, liens fractions↔proportionnalité, comparaisons fines, situations de partage complexes."),
    "B": ("Consolidation (solides)", "Vert", "#2ECC71",
          "Varier les contextes, automatiser sans perdre le sens : exercices courts, verbalisation de méthode, mini-quiz."),
    "C": ("Renforcement opérations", "Jaune", "#F1C40F",
          "Ciblage opérations : équivalences, entraînement guidé + erreurs typiques."),
    "D": ("Soutien ciblé", "Orange", "#E67E22",
          "Revoir procédures une par une : schémas, étapes explicites, correction commentée, transfert progressif."),
    "E": ("Remédiation représentation (sens)", "Rouge", "#E74C3C",
          "Reprendre le sens : part d’un tout, bande/disque, lecture/écriture, comparaison visuelle, équivalences par découpage."),
    "F": ("Remédiation intensive", "Violet", "#6C3483",
          "Accompagnement rapproché : micro-objectifs, manipulations, consignes courtes, rituels, formative très fréquente."),
}

# ───────────────────────────── LLM / RAG ─────────────────────────
def _safe_llm() -> Optional[ChatOpenAI]:
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=OPENAI_API_KEY,
    )

llm = _safe_llm()

from rag_langchain import rag_chain  # renvoie {"answer": str, "source_documents": [...]}

# ───────────────────────────── Helpers sources ───────────────────
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

    if doc_type == "txt":
        return f"{src_name}"

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

# ───────────────────────────── Core "fractions" ──────────────────
def fractions_rag(question: str) -> str:
    if not DOCS_DIR.exists():
        return f"❌ Dossier corpus introuvable: {DOCS_DIR}"

    # ✅ Fallback HF : si PDF absent, mais TXT présent → OK
    txt_candidates = list(DOCS_DIR.glob("*.txt"))
    has_pdf = PDF_PATH.exists()
    has_txt = len(txt_candidates) > 0

    if not has_pdf and has_txt:
        demo_note = "ℹ️ Corpus TXT utilisé (mode démo Hugging Face : PDF non embarqué)."
    else:
        demo_note = ""

    if not has_pdf and not has_txt:
        return (
            "❌ Corpus introuvable.\n"
            f"- PDF attendu: {PDF_PATH}\n"
            f"- ou au moins un .txt dans: {DOCS_DIR}\n"
        )

    if not OPENAI_API_KEY:
        return (
            "❌ OPENAI_API_KEY manquant.\n"
            "➡️ Sur Hugging Face : Settings → Variables and secrets → Secrets → Add secret\n"
            "   OPENAI_API_KEY = ta clé\n"
        )

    result = rag_chain({"question": question})
    answer = (result.get("answer") or "").strip() or "Je ne sais pas."
    sources = result.get("source_documents") or []

    prefix = (demo_note + "\n\n") if demo_note else ""
    return f"{prefix}{answer}\n\n{_sources_block(sources)}"


def didactic_check(text: str) -> str:
    if llm is None:
        return (
            "❌ OPENAI_API_KEY manquant : impossible d'améliorer didactiquement.\n"
            "➡️ Ajoute la clé dans les secrets du Space."
        )

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


def lookup_error_remediation(error_id: str) -> str:
    eid = (error_id or "").strip()
    if not eid:
        return "❌ Donne un error_id (ex: add_denominators)."

    err_df = _load_excel(ERREURS_XLSX)
    rem_df = _load_excel(REMED_XLSX)

    if err_df.empty and rem_df.empty:
        return (
            "ℹ️ Mode démo : pas d’Excel erreurs/remédiations embarqués.\n"
            f"- Attendu: {ERREURS_XLSX.name} et/ou {REMED_XLSX.name}\n"
            "➡️ Tu peux les ajouter plus tard (ou les convertir en .txt)."
        )

    err_df.columns = [c.strip().lower() for c in err_df.columns]
    rem_df.columns = [c.strip().lower() for c in rem_df.columns]

    if "error_id" not in err_df.columns and "error_id" not in rem_df.columns:
        return "❌ Les fichiers Excel doivent contenir une colonne `error_id`."

    err = err_df[err_df.get("error_id", pd.Series()).astype(str).str.strip() == eid] if not err_df.empty else pd.DataFrame()
    rem = rem_df[rem_df.get("error_id", pd.Series()).astype(str).str.strip() == eid] if not rem_df.empty else pd.DataFrame()

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

# ───────────────────────────── Analyse classe ────────────────────
@dataclass
class AnalyzeResult:
    df_students: pd.DataFrame
    stats_objectifs: pd.DataFrame
    recos_groupes: pd.DataFrame
    text_summary: str
    source_path: Path
    used_sample: bool


def _resolve_path(p: str) -> Tuple[Path, bool]:
    """Retourne (path, used_sample)."""
    p = (p or "").strip().strip('"').strip("'")

    if p:
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        return path, False

    if RESPONSES_CSV.exists():
        return RESPONSES_CSV, False

    if SAMPLE_RESPONSES_CSV.exists():
        return SAMPLE_RESPONSES_CSV, True

    return RESPONSES_CSV, False


def _read_csv_any(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _ensure_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, OBJ_COUNT + 1):
        col = f"OBJ{i}_Score"
        if col not in df.columns:
            raise ValueError(
                f"Colonne manquante: {col}. "
                "Ajoute OBJx_Score (0/1) pour chaque objectif."
            )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    if "Total_Score" not in df.columns:
        score_cols = [f"OBJ{i}_Score" for i in range(1, OBJ_COUNT + 1)]
        df["Total_Score"] = df[score_cols].sum(axis=1)
    return df


def _compute_profile_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for prof, obj_list in PROFILE_MAP.items():
        cols = [f"OBJ{i}_Score" for i in obj_list if f"OBJ{i}_Score" in df.columns]
        df[prof] = df[cols].sum(axis=1) if cols else 0
    return df


def _objective_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for i in range(1, OBJ_COUNT + 1):
        col = f"OBJ{i}_Score"
        ok = int(df[col].sum())
        ko = n - ok
        pct = (ok / n * 100.0) if n else 0.0
        rows.append({"Objectif": f"OBJ{i}", "Reussites": ok, "Echecs": ko, "Taux_reussite_%": round(pct, 1)})
    stats = pd.DataFrame(rows).sort_values("Taux_reussite_%", ascending=True).reset_index(drop=True)
    return stats


def _pick_group(row: pd.Series) -> str:
    total = int(row.get("Total_Score", 0))
    rep = int(row.get("Rep_Score", 0))
    ops = int(row.get("Ops_Score", 0))

    if total <= 1:
        return "F"
    if total <= 4 and rep <= 0:
        return "E"
    if total <= 2 and ops <= 1:
        return "F"
    if total >= THRESH_A:
        return "A"
    if total >= THRESH_B:
        return "B"
    if total >= THRESH_C and ops <= 2:
        return "C"
    if total >= THRESH_D:
        return "D"
    return "E"


def _add_group_info(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Groupe"] = df.apply(_pick_group, axis=1)

    def info(g: str):
        name, color_name, color_hex, reco = GROUP_INFO[g]
        return pd.Series([name, color_name, color_hex, reco])

    df[["Groupe_Label", "Couleur", "Couleur_HEX", "Reco_Pedago"]] = df["Groupe"].apply(info)
    return df


def analyze_class(path_str: str = "") -> AnalyzeResult:
    path, used_sample = _resolve_path(path_str)
    df = _read_csv_any(path)

    df = _ensure_score_columns(df)
    df = _compute_profile_scores(df)
    df = _add_group_info(df)

    stats = _objective_stats(df)
    top_hard = stats.sort_values("Taux_reussite_%", ascending=True).head(TOP_HARD)

    grp = (
        df.groupby(["Groupe", "Groupe_Label", "Couleur", "Couleur_HEX", "Reco_Pedago"])
        .size()
        .reset_index(name="Effectif")
        .sort_values(["Groupe"])
        .reset_index(drop=True)
    )

    lines = []
    if used_sample:
        lines.append("ℹ️ Analyse réalisée sur un échantillon anonymisé (mode démo HF).\n")

    lines.append("📊 Analyse par objectif – Fractions 5e")
    for i in range(1, OBJ_COUNT + 1):
        row = stats[stats["Objectif"] == f"OBJ{i}"].iloc[0]
        lines.append(
            f"{row['Objectif']} : {row['Taux_reussite_%']} % "
            f"({row['Reussites']} réussites / {row['Echecs']} échecs)"
        )

    lines.append(f"\n⚠️ Objectifs les plus difficiles (top {TOP_HARD})")
    for _, r in top_hard.iterrows():
        lines.append(f"{r['Objectif']} – {r['Taux_reussite_%']} %")

    lines.append("\n👥 Groupes de besoin (score + profil)")
    for g in ["A", "B", "C", "D", "E", "F"]:
        if g not in set(grp["Groupe"]):
            continue
        sub = grp[grp["Groupe"] == g].iloc[0]
        name, color_name, color_hex, reco = GROUP_INFO[g]
        emoji = "🟢" if g == "A" else "🟩" if g == "B" else "🟨" if g == "C" else "🟧" if g == "D" else "🟥" if g == "E" else "🟪"
        lines.append(
            f"{emoji} Groupe {g} – {name} ({int(sub['Effectif'])} élèves) — Couleur: {color_name} ({color_hex})\n"
            f"→ Reco: {reco}"
        )

    return AnalyzeResult(
        df_students=df,
        stats_objectifs=stats.sort_values("Objectif").reset_index(drop=True),
        recos_groupes=grp,
        text_summary="\n".join(lines),
        source_path=path,
        used_sample=used_sample,
    )


def export_analysis(last_path: str = "") -> str:
    res = analyze_class(last_path)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    p1 = (EXPORTS_DIR / "stats_objectifs.csv").resolve()
    p2 = (EXPORTS_DIR / "groupes_eleves.csv").resolve()
    p3 = (EXPORTS_DIR / "recommandations_groupes.csv").resolve()

    res.stats_objectifs.to_csv(p1, index=False, encoding="utf-8-sig")
    res.df_students.to_csv(p2, index=False, encoding="utf-8-sig")
    res.recos_groupes.to_csv(p3, index=False, encoding="utf-8-sig")

    return (
        "✅ Export terminé (3 fichiers créés dans exports/) :\n"
        f"{p1}\n{p2}\n{p3}"
    )

# ───────────────────────────── Command router ────────────────────
def _extract_error_id(text: str) -> Optional[str]:
    m = re.search(r"error_id\s*[:=]\s*([a-z0-9_\-]+)", text.strip(), flags=re.I)
    return m.group(1) if m else None


def _is_didactic_request(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in [
        "rends didactique", "version didactique", "reformule", "vulgarise", "explique avec du sens"
    ])


def _parse_slash_command(msg: str) -> Tuple[str, str]:
    tokens = shlex.split(msg)
    cmd = tokens[0].lower() if tokens else ""
    arg = " ".join(tokens[1:]).strip() if len(tokens) > 1 else ""
    return cmd, arg


_LAST_ANALYZE_PATH: str = ""


def run_agent(message: str) -> str:
    global _LAST_ANALYZE_PATH

    msg = (message or "").strip()
    if not msg:
        return "Écris une question 🙂"

    if msg.startswith("/"):
        cmd, arg = _parse_slash_command(msg)

        if cmd == "/help":
            return (
                "📌 Commandes disponibles\n"
                "- /help\n- /examples\n- /analyze\n- /analyze <chemin_fichier>\n- /export\n"
            )

        if cmd == "/examples":
            return (
                "🧪 Exemples\n"
                "- Résume les opérations sur les fractions\n"
                "- Pourquoi 1/2 + 1/3 = 2/5 est faux ?\n"
                "- Rends didactique : mettre au même dénominateur\n"
                "- /analyze\n- /export\n"
            )

        if cmd == "/analyze":
            try:
                _LAST_ANALYZE_PATH = arg or ""
                res = analyze_class(arg or "")
                return res.text_summary
            except Exception as e:
                return (
                    "⚠️ Impossible d’exécuter la commande.\n"
                    f"Détail: {type(e).__name__}: {e}\n"
                    "\n✅ Solutions rapides :\n"
                    "- Vérifie que `OBJ1_Score ... OBJ10_Score` existent (0/1)\n"
                    "- Vérifie le chemin du fichier si tu utilises /analyze <chemin>\n"
                    "- Sur HF : ajoute sample_responses.csv pour le mode démo\n"
                )

        if cmd == "/export":
            try:
                return export_analysis(_LAST_ANALYZE_PATH)
            except Exception as e:
                return (
                    "⚠️ Impossible d’exécuter /export.\n"
                    f"Détail: {type(e).__name__}: {e}"
                )

        return "❓ Commande inconnue. Tape /help."

    eid = _extract_error_id(msg)
    if eid:
        return lookup_error_remediation(eid)

    if _is_didactic_request(msg):
        return didactic_check(msg)

    try:
        return fractions_rag(msg)
    except Exception as e:
        return (
            "⚠️ Désolé, je n’ai pas pu générer de réponse.\n\n"
            f"Erreur: {type(e).__name__}: {e}"
        )
