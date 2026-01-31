from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# 0) Modèle & constantes
# ─────────────────────────────────────────────────────────────────────

GROUPS = {
    "A": {"label": "Approfondissement (experts)", "color": "🟣", "hex": "#8E44AD"},
    "B": {"label": "Consolidation (solides)", "color": "🟢", "hex": "#27AE60"},
    "C": {"label": "Renforcement opérations", "color": "🟠", "hex": "#E67E22"},
    "D": {"label": "Soutien ciblé", "color": "🟡", "hex": "#F1C40F"},
    "E": {"label": "Remédiation représentation", "color": "🔵", "hex": "#2980B9"},
    "F": {"label": "Remédiation intensive", "color": "🔴", "hex": "#C0392B"},
}


DEFAULT_OBJECTIVE_META = {
    # Tu peux renommer selon ton programme (OBJ1..OBJ8 ici)
    "OBJ1": {"axis": "representation", "title": "Sens de la fraction (partage / unité)"},
    "OBJ2": {"axis": "representation", "title": "Lecture/écriture de fractions"},
    "OBJ3": {"axis": "representation", "title": "Comparer / ordonner des fractions"},
    "OBJ4": {"axis": "representation", "title": "Représentation (droite graduée / partages)"},
    "OBJ5": {"axis": "operations", "title": "Addition / soustraction de fractions"},
    "OBJ6": {"axis": "operations", "title": "Multiplication de fractions"},
    "OBJ7": {"axis": "operations", "title": "Division de fractions / inverse"},
    "OBJ8": {"axis": "operations", "title": "Problèmes & choix d’opération"},
}


@dataclass
class AnalyticsResult:
    df: pd.DataFrame
    objective_stats: pd.DataFrame
    student_profiles: pd.DataFrame
    group_summary: pd.DataFrame


# ─────────────────────────────────────────────────────────────────────
# 1) Lecture & préparation
# ─────────────────────────────────────────────────────────────────────

def read_student_file(path: Path) -> pd.DataFrame:
    """
    Lit CSV/TSV/XLSX.
    Attend au minimum les colonnes:
    ID_Eleve, Nom, Prenom, Classe, OBJ1_Question, OBJ1_Reponse, ..., OBJ8_Question, OBJ8_Reponse
    OU (mieux) OBJ1_Score..OBJ8_Score.
    """
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")

    suffix = path.suffix.lower()
    if suffix in [".csv"]:
        df = pd.read_csv(path)
    elif suffix in [".tsv"]:
        df = pd.read_csv(path, sep="\t")
    elif suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Format non supporté. Utilise .csv, .tsv ou .xlsx")

    df.columns = [c.strip() for c in df.columns]
    return df


def _objective_ids_from_df(df: pd.DataFrame) -> List[str]:
    objs = []
    for i in range(1, 21):  # marge si tu ajoutes OBJ9..OBJ10 etc
        oid = f"OBJ{i}"
        if f"{oid}_Reponse" in df.columns or f"{oid}_Score" in df.columns:
            objs.append(oid)
    return objs


def ensure_scores(
    df: pd.DataFrame,
    answer_key_path: Optional[Path] = None,
    score_col_suffix: str = "_Score",
) -> pd.DataFrame:
    """
    Crée/valide les colonnes OBJx_Score.
    Priorité :
      1) si OBJx_Score existe → on garde.
      2) sinon, si answer_key.json fourni → scoring déterministe sur la réponse.
      3) sinon → lève une erreur claire (pas de scoring possible).
    """
    df = df.copy()
    obj_ids = _objective_ids_from_df(df)

    missing_scores = [oid for oid in obj_ids if f"{oid}{score_col_suffix}" not in df.columns]
    if not missing_scores:
        # normalisation (0/1 int)
        for oid in obj_ids:
            c = f"{oid}{score_col_suffix}"
            df[c] = df[c].fillna(0).astype(int).clip(0, 1)
        return df

    if answer_key_path is None:
        raise ValueError(
            "Il manque des colonnes OBJx_Score et aucun answer_key.json n’a été fourni.\n"
            "➡️ Solution: ajoute OBJ1_Score..OBJn_Score, OU fournis un answer_key.json."
        )
    if not answer_key_path.exists():
        raise FileNotFoundError(f"answer_key.json introuvable: {answer_key_path}")

    key = json.loads(answer_key_path.read_text(encoding="utf-8"))
    # key attendu: {"OBJ1": {"accepted": ["1/2","0.5"], "regex": ["^...$"]}, ...}

    for oid in obj_ids:
        score_col = f"{oid}{score_col_suffix}"
        if score_col in df.columns:
            df[score_col] = df[score_col].fillna(0).astype(int).clip(0, 1)
            continue

        if f"{oid}_Reponse" not in df.columns:
            df[score_col] = 0
            continue

        rules = key.get(oid, {})
        accepted = [str(x).strip() for x in rules.get("accepted", [])]
        regexes = [str(x).strip() for x in rules.get("regex", [])]

        def grade(x: object) -> int:
            s = "" if pd.isna(x) else str(x).strip()
            if not s:
                return 0
            if accepted and s in accepted:
                return 1
            for r in regexes:
                try:
                    if pd.Series([s]).str.contains(r, regex=True, case=False).iloc[0]:
                        return 1
                except Exception:
                    continue
            return 0

        df[score_col] = df[f"{oid}_Reponse"].apply(grade).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────
# 2) Analyse par objectif
# ─────────────────────────────────────────────────────────────────────

def objective_success_stats(df: pd.DataFrame, obj_ids: List[str], meta: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for oid in obj_ids:
        score_col = f"{oid}_Score"
        if score_col not in df.columns:
            continue
        ok = int(df[score_col].sum())
        rate = (ok / n) if n else 0.0
        rows.append({
            "objectif": oid,
            "axe": meta.get(oid, {}).get("axis", "?"),
            "intitule": meta.get(oid, {}).get("title", ""),
            "reussite": ok,
            "effectif": n,
            "taux_reussite": round(rate * 100, 1),
        })
    out = pd.DataFrame(rows).sort_values("taux_reussite")
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────
# 3) Profils & Groupes (A..F)
# ─────────────────────────────────────────────────────────────────────

def build_profiles_and_groups(df: pd.DataFrame, obj_ids: List[str], meta: Dict[str, Dict]) -> pd.DataFrame:
    """
    Logique stable, pédagogique, explicable :
      - rep = moyenne des OBJ axis=representation
      - ops = moyenne des OBJ axis=operations
      - total = moyenne globale
    Groupes :
      A : total >= 0.90
      B : total >= 0.75
      C : ops < 0.60 et rep >= 0.60
      E : rep < 0.60 et ops >= 0.60
      F : rep < 0.40 et ops < 0.40
      D : sinon (soutien ciblé)
    """
    rep_objs = [oid for oid in obj_ids if meta.get(oid, {}).get("axis") == "representation" and f"{oid}_Score" in df.columns]
    ops_objs = [oid for oid in obj_ids if meta.get(oid, {}).get("axis") == "operations" and f"{oid}_Score" in df.columns]
    all_objs = [oid for oid in obj_ids if f"{oid}_Score" in df.columns]

    out = df.copy()

    def mean_score(row, oids):
        if not oids:
            return 0.0
        vals = [row[f"{oid}_Score"] for oid in oids]
        return float(sum(vals)) / float(len(vals))

    out["rep_score"] = out.apply(lambda r: mean_score(r, rep_objs), axis=1)
    out["ops_score"] = out.apply(lambda r: mean_score(r, ops_objs), axis=1)
    out["total_score"] = out.apply(lambda r: mean_score(r, all_objs), axis=1)

    def group_of(r) -> str:
        total = r["total_score"]
        rep = r["rep_score"]
        ops = r["ops_score"]

        if total >= 0.90:
            return "A"
        if total >= 0.75:
            return "B"
        if rep < 0.40 and ops < 0.40:
            return "F"
        if ops < 0.60 and rep >= 0.60:
            return "C"
        if rep < 0.60 and ops >= 0.60:
            return "E"
        return "D"

    out["groupe"] = out.apply(group_of, axis=1)
    out["groupe_couleur"] = out["groupe"].map(lambda g: GROUPS[g]["color"])
    out["groupe_label"] = out["groupe"].map(lambda g: GROUPS[g]["label"])

    # colonnes essentielles (sans casser ton fichier)
    wanted = ["ID_Eleve", "Nom", "Prenom", "Classe", "rep_score", "ops_score", "total_score", "groupe", "groupe_couleur", "groupe_label"]
    existing = [c for c in wanted if c in out.columns]
    return out[existing].copy()


# ─────────────────────────────────────────────────────────────────────
# 4) Synthèse par groupe + recommandations
# ─────────────────────────────────────────────────────────────────────

def group_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.empty:
        return pd.DataFrame()

    rows = []
    for g, sub in profiles.groupby("groupe"):
        rows.append({
            "groupe": g,
            "couleur": GROUPS[g]["color"],
            "label": GROUPS[g]["label"],
            "effectif": int(len(sub)),
            "rep_moy": round(float(sub["rep_score"].mean()) * 100, 1),
            "ops_moy": round(float(sub["ops_score"].mean()) * 100, 1),
            "total_moy": round(float(sub["total_score"].mean()) * 100, 1),
            "recommandations": recommendations_for_group(g),
        })
    return pd.DataFrame(rows).sort_values("groupe").reset_index(drop=True)


def recommendations_for_group(g: str) -> str:
    # Texte court, actionnable (tu pourras l’enrichir dans ton corpus Excel Remédiations)
    if g == "A":
        return "Défis: problèmes ouverts, justifications, fractions/ratio, tâches complexes, pair teaching."
    if g == "B":
        return "Consolidation: automatismes + justification; mini-défis; erreurs pièges contrôlées."
    if g == "C":
        return "Renforcement opérations: dénominateur commun, sens des opérations, exercices gradués, feedback immédiat."
    if g == "D":
        return "Soutien ciblé: reprendre 1–2 objectifs précis, ateliers courts, verbalisation, exemples/contre-exemples."
    if g == "E":
        return "Représentation: schémas (partages), droite graduée, manipulations, liens fraction↔décimal↔mesure."
    if g == "F":
        return "Intensif: micro-objectifs, manipulations, guidage fort, répétition espacée, évaluations flash."
    return "—"


# ─────────────────────────────────────────────────────────────────────
# 5) Orchestration globale + export
# ─────────────────────────────────────────────────────────────────────

def run_full_analytics(
    file_path: Path,
    answer_key_path: Optional[Path] = None,
    objective_meta: Optional[Dict[str, Dict]] = None,
) -> AnalyticsResult:
    meta = objective_meta or DEFAULT_OBJECTIVE_META
    df = read_student_file(file_path)
    obj_ids = _objective_ids_from_df(df)

    df_scored = ensure_scores(df, answer_key_path=answer_key_path)
    stats = objective_success_stats(df_scored, obj_ids, meta)
    profiles = build_profiles_and_groups(df_scored, obj_ids, meta)
    summary = group_summary(profiles)

    return AnalyticsResult(
        df=df_scored,
        objective_stats=stats,
        student_profiles=profiles,
        group_summary=summary,
    )


def export_results(res: AnalyticsResult, out_dir: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "objective_stats.csv"
    p2 = out_dir / "student_profiles.csv"
    p3 = out_dir / "group_summary.csv"

    res.objective_stats.to_csv(p1, index=False)
    res.student_profiles.to_csv(p2, index=False)
    res.group_summary.to_csv(p3, index=False)

    return p1, p2, p3
