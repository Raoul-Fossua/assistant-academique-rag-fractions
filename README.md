#1️⃣ Assistant Académique + RAG (Fractions en classe de 5e)

Assistant pédagogique intelligent basé sur une architecture **RAG (Retrieval-Augmented Generation)**, 
conçu pour l’enseignement des **fractions en classe de 5e**.

Le projet combine :
- **LangChain** (orchestration RAG + agents),
- **Chainlit** (interface conversationnelle),
- **FAISS** (vectorisation locale),
- des **documents pédagogiques réels** (cours, erreurs fréquentes, remédiations).

Projet développé dans le cadre du **DU Sorbonne Data Analytics**.

##2️⃣ 🎯 Objectifs pédagogiques

- Répondre aux questions de cours sur les fractions (niveau 5e) **uniquement à partir du corpus**
- Identifier et expliquer les **erreurs fréquentes** des élèves
- Proposer des **remédiations didactiques structurées**
- Préparer une future **analyse de profils d’erreurs** (clustering d’élèves)
- Éviter toute réponse “hors programme” ou inventée

##3️⃣ 🧠 Architecture technique

- **LLM** : OpenAI (via `langchain-openai`)
- **RAG** :
  - PDF de cours (Fractions 5e)
  - Fichiers Excel (erreurs fréquentes + remédiations)
- **Vector store** : FAISS (local)
- **Interface** : Chainlit
- **Langage** : Python 3.11

Pipeline : Documents → Chunking → Embeddings → FAISS → Retriever → LLM → Réponse sourcée

##4️⃣ ⚙️ Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## 5️⃣ Configuration (.env)  

```md
## 🔐 Configuration

Créer un fichier `.env` à la racine du projet (non versionné) :

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
FAISS_DIR=C:\faiss_fractions

##6️⃣ Données (section cruciale RGPD / droits)

```md
## 📁 Données

Les données ne sont **pas versionnées** sur GitHub.

Arborescence attendue :

- `data/Corpus/`
  - `Cours_Fractions_5e.pdf`
  - `Erreurs_Fractions_5e.xlsx`
  - `Remediations_Fractions_5e.xlsx`
- `data/Students/`
  - `responses.csv` (données anonymisées)

⚠️ Les fichiers pédagogiques et les données élèves restent **strictement locales**.


##7️⃣ ▶️ Lancer l’application

```powershell
python -m chainlit run .\chainlit_app.py -w

---

##8️⃣ État du projet  

```md
## 🚧 État du projet

- RAG fonctionnel (PDF + Excel)
- Agent pédagogique opérationnel
- Interface Chainlit stable

##9️⃣ 🔭 Perspectives (vision long terme)
- Clustering automatique des profils d’erreurs
- Tableaux de bord enseignants
- Extension à d’autres chapitres (proportionnalité, géométrie…)


## 👤 Auteur

Raoul FOSSUA TINDO  

Enseignant en mathématiques – Projet développé dans le cadre du DU Sorbonne Data Analytics (Paris 1 Panthéon-Sorbonne)
