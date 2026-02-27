---
title: "📚 Assistant pédagogique intelligent – Fractions (5e)"
emoji: "📚"
colorFrom: "blue"
colorTo: "green"
sdk: "docker"
pinned: false
---

# 📚 Assistant pédagogique intelligent – Fractions (5e)
**Architecture RAG + Analyse de données éducatives (Chainlit)**  
Projet DU Sorbonne Data Analytics — IA générative appliquée à l’enseignement des fractions en classe de 5e.

---

## 🎓 Cadre académique
Ce projet est réalisé dans le cadre du **DU Sorbonne Data Analytics (Université Paris 1 Panthéon-Sorbonne)**.

**Objectif :** concevoir un assistant pédagogique intelligent combinant :
- **IA générative** (LLM)
- **RAG** (Retrieval-Augmented Generation) avec **traçabilité des sources**
- **Analyse de données éducatives** (diagnostic, groupes de besoin, exports)

Application : **fractions – niveau 5e**.

---

## 🎯 Problématique pédagogique
L’enseignement des fractions au collège est souvent marqué par :
- des **erreurs conceptuelles récurrentes** (sens du dénominateur, équivalences, opérations),
- une difficulté à relier **procédure** et **compréhension**,
- un besoin fort de **différenciation**, idéalement appuyée sur des données d’élèves.

👉 **Problème central :**  
Comment exploiter des données élèves et des ressources pédagogiques pour **diagnostiquer finement** les difficultés, **former des groupes de besoin**, et **proposer des recommandations pédagogiques**, tout en garantissant la **traçabilité** des réponses ?

---

## 🧠 Objectifs du projet

### Objectifs pédagogiques
- Expliquer les notions sur les fractions **avec du sens** (pas de “règles magiques”).
- Identifier les **erreurs fréquentes** et leur origine didactique.
- Produire des **recommandations pédagogiques** ciblées (par profils d’élèves).

### Objectifs data & IA
- Mettre en œuvre une architecture **RAG fiable** (sources citées).
- Exploiter un fichier de réponses élèves pour :
  - analyser les **taux de réussite** par objectif (OBJ1 → OBJ10),
  - identifier les **objectifs difficiles**,
  - regrouper les élèves en **groupes de besoin**.
- Générer des **exports CSV** exploitables par l’enseignant.

---

## 🏗️ Architecture générale (vue d’ensemble)

Assistant pédagogique
│
├── RAG pédagogique (LangChain + FAISS)
│ ├── Corpus TXT (mode démo HF)
│ ├── PDF : cours fractions (optionnel / local)
│ ├── Excel : erreurs fréquentes (optionnel)
│ └── Excel : remédiations (optionnel)
│
├── Analyse de données élèves (Pandas)
│ ├── Scores par objectif OBJ1..OBJ10 (0/1)
│ ├── Statistiques de réussite
│ ├── Profils (Rep / Compare / Equiv / Ops)
│ └── Groupes de besoin A..F
│
└── Interface Chainlit (enseignant)

---

## 🧾 Données utilisées

### 1) Corpus pédagogique (RAG)
Objectif : produire des réponses **ancrées dans un corpus** et **citées**.

- `data/Corpus/corpus_fractions_5e.txt` ✅ **recommandé pour Hugging Face**
- `data/Corpus/Cours_Fractions_5e.pdf` (local, non versionné en général)
- `data/Corpus/Erreurs_Fractions_5e.xlsx` (optionnel)
- `data/Corpus/Remediations_Fractions_5e.xlsx` (optionnel)

📌 L’assistant doit **refuser d’inventer** :  
si l’info n’est pas dans le corpus → **“Je ne sais pas.”**

---

### 2) Données élèves (responses.csv)
Fichier attendu (structure minimale) :
- `OBJ1_Score ... OBJ10_Score` (scores binaires 0/1)
- optionnel : Nom/Prénom/Classe (souvent anonymisé)

📌 Sur Hugging Face : mode démo via :
- `data/Students/sample_responses.csv` ✅ (anonymisé, petit, présentable)

---

## 📊 Analyse pédagogique automatisée

### Analyse par objectif
- Calcul du **taux de réussite** par objectif
- Identification automatique des **objectifs les plus difficiles**

### Groupes de besoin (6 profils)
| Groupe | Profil | Couleur | Finalité pédagogique |
|------:|--------|---------|----------------------|
| A | Approfondissement (experts) | Vert foncé | Défis, justification |
| B | Consolidation | Vert | Stabiliser les acquis |
| C | Renforcement opérations | Jaune | Entraînement ciblé |
| D | Soutien ciblé | Orange | Procédures guidées |
| E | Remédiation sens | Rouge | Représentations |
| F | Remédiation intensive | Violet | Accompagnement rapproché |

Chaque groupe est associé à :
- une couleur,
- une recommandation pédagogique explicite.

---

## 📤 Exports générés
Commande `/export` :

| Fichier | Contenu |
|--------|---------|
| `exports/stats_objectifs.csv` | Taux de réussite par objectif |
| `exports/groupes_eleves.csv` | Groupe, couleur, score par élève |
| `exports/recommandations_groupes.csv` | Synthèse pédagogique par groupe |

➡️ Exploitables directement en **différenciation**, **APC**, **conseil de cycle**.

---

## 💬 Interface utilisateur (Chainlit)

### Commandes disponibles
- `/help` – aide rapide
- `/examples` – exemples de questions
- `/analyze` – analyse de classe (fichier par défaut)
- `/analyze <chemin>` – analyse d’un autre fichier
- `/export` – génération des CSV pédagogiques

### Comportement attendu
- support des entrées multi-lignes
- gestion des erreurs sans crash
- aucune hallucination : **sources ou “je ne sais pas”**

---

## 🚀 Démarrage local

### 1) Installer
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

