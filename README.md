📚 Assistant pédagogique intelligent – Fractions (5e)
Architecture RAG + Analyse de données éducatives
🎓 Cadre académique

Projet réalisé dans le cadre du DU Sorbonne Data Analytics
Objectif : concevoir un assistant pédagogique intelligent combinant
IA générative, RAG (Retrieval-Augmented Generation) et analyse de données éducatives, appliqué à l’enseignement des fractions en classe de 5e.

🎯 Problématique pédagogique

L’enseignement des fractions en collège est marqué par :

des erreurs conceptuelles récurrentes (sens du dénominateur, équivalences, opérations),

une difficulté à relier procédures et compréhension,

un besoin fort de différenciation pédagogique à partir de données réelles d’élèves.

👉 Problème central :

Comment exploiter des données élèves et des ressources pédagogiques pour diagnostiquer finement les difficultés, former des groupes de besoin, et proposer des recommandations pédagogiques adaptées, tout en garantissant la traçabilité des réponses ?

🧠 Objectifs du projet
Objectifs pédagogiques

Expliquer les notions sur les fractions avec du sens (pas de règles magiques).

Identifier les erreurs fréquentes et leurs causes didactiques.

Produire des recommandations pédagogiques ciblées par profil d’élèves.

Objectifs data & IA

Mettre en œuvre une architecture RAG fiable (sources traçables).

Exploiter un fichier de réponses élèves pour :

analyser les taux de réussite par objectif,

identifier les objectifs les plus difficiles,

classifier les élèves en groupes de besoin.

Générer des exports exploitables par l’enseignant (CSV).

🏗️ Architecture générale
Assistant pédagogique
│
├── RAG pédagogique (LangChain)
│   ├── PDF : cours sur les fractions
│   ├── Excel : erreurs fréquentes
│   └── Excel : remédiations pédagogiques
│
├── Analyse de données élèves
│   ├── Scores par objectif (OBJ1 → OBJ10)
│   ├── Statistiques de réussite
│   ├── Profils d’erreurs
│   └── Groupes de besoin
│
├── IA générative (LLM)
│   ├── Explications contextualisées
│   ├── Reformulation didactique
│   └── Recommandations pédagogiques
│
└── Interface Chainlit (enseignant)

🧾 Données utilisées
1️⃣ Corpus pédagogique (RAG)

Cours_Fractions_5e.pdf

Erreurs_Fractions_5e.xlsx

Remediations_Fractions_5e.xlsx

👉 Ces documents sont interrogés par le modèle, et toute réponse cite explicitement ses sources.

2️⃣ Données élèves (responses.csv)

Structure attendue :

ID_Eleve | Nom | Prenom | Classe
OBJ1_Score ... OBJ10_Score
Total_Score
Rep_Score | Compare_Score | Equiv_Score | Ops_Score


Scores binaires (0/1) par objectif d’apprentissage

Données anonymisables et non versionnées (RGPD)

📊 Analyse pédagogique automatisée
Analyse par objectif

Calcul du taux de réussite par objectif

Identification automatique des objectifs les plus difficiles

Groupes de besoin (6 profils)
Groupe	Profil	Couleur	Finalité pédagogique
A	Approfondissement (experts)	Vert foncé	Défis, justification
B	Consolidation	Vert	Stabiliser les acquis
C	Renforcement opérations	Jaune	Entraînement ciblé
D	Soutien ciblé	Orange	Procédures guidées
E	Remédiation sens	Rouge	Représentations
F	Remédiation intensive	Violet	Accompagnement rapproché

Chaque groupe est associé à :

une couleur,

un profil d’erreurs dominant,

une recommandation pédagogique explicite.

📤 Exports générés

Commande /export :

Fichier	Contenu
stats_objectifs.csv	Taux de réussite par objectif
groupes_eleves.csv	Groupe, couleur, score par élève
recommandations_groupes.csv	Synthèse pédagogique par groupe

👉 Exploitables directement en conseil de cycle, APC ou différenciation.

💬 Interface utilisateur (Chainlit)

Commandes disponibles :

/help – aide rapide

/examples – exemples de questions

/analyze – analyse de la classe (fichier par défaut)

/analyze <chemin> – analyse d’un autre fichier

/export – génération des CSV pédagogiques

L’assistant :

gère les entrées multi-lignes,

ne plante jamais (gestion des erreurs),

refuse d’inventer si l’information n’est pas disponible.

🔐 Sécurité & éthique

Données élèves non versionnées (.gitignore)

Clés API sécurisées (.env)

Séparation claire entre :

code,

données,

résultats générés

Respect des principes RGPD et de la propriété intellectuelle

🛠️ Technologies utilisées

Python 3.11

LangChain (RAG)

FAISS (vectorisation locale)

OpenAI API (LLM)

Pandas / NumPy

Chainlit (interface pédagogique)

🚀 Perspectives d’évolution

Ajout de clustering automatique (KMeans, silhouette)

Suivi longitudinal des élèves

Extension à d’autres chapitres (proportionnalité, nombres relatifs)

Interface enseignant enrichie (tableaux de bord)

👨‍🏫 Public cible

<< HEAD
Enseignants de mathématiques collège
=======
Enseignants de mathématiques Collège / Lycée
>> 14ab861 (docs: add academic README (DU Sorbonne Data Analytics))

Chercheurs en didactique des mathématiques

Encadrants data / IA éducative

📌 Conclusion

<< HEAD
Ce projet illustre comment l’IA générative, lorsqu’elle est contrainte par des sources et pilotée par les données, peut devenir un véritable outil pédagogique, au service de la compréhension des élèves et de la décision didactique de l’enseignant.
=======
Ce projet illustre comment l’IA générative, lorsqu’elle est contrainte par des sources et pilotée par les données, peut devenir un véritable outil pédagogique, au service de la compréhension des élèves et de la décision didactique de l’enseignant.
>> 14ab861 (docs: add academic README (DU Sorbonne Data Analytics))
