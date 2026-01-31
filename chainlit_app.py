from __future__ import annotations

import shlex
from typing import List, Tuple

import chainlit as cl

# agents.py doit exposer:
#   run_agent(message: str) -> str
from agents import run_agent


WELCOME = """👋 Bonjour ! **Assistant pédagogique intelligent – Fractions (5e)**

Je peux :
- 📘 Expliquer une notion / une méthode sur les fractions (RAG + sources)
- ✍️ Reformuler une explication en version didactique
- 🧩 Analyser une classe (objectifs + groupes + recommandations)

Commandes utiles :
- `/help`
- `/examples`
- `/analyze` (analyse classe depuis le fichier par défaut)
- `/analyze <chemin_fichier>` (si tu veux pointer un autre fichier)
- `/export` (génère 3 CSV dans `exports/`)
"""


HELP = """📌 **Aide rapide**

✅ Exemples :
- *Résume les opérations sur les fractions.*
- *Explique “mettre au même dénominateur” avec du sens.*
- *Pourquoi certains élèves font 1/2 + 1/3 = 2/5 ?*
- *Rends didactique : “on met au même dénominateur”.*
- *Analyse ma classe (responses.csv).*

🧾 Sources :
Quand je réponds via le corpus, j’ajoute toujours un bloc **Sources** (PDF/pages, Excel, etc.).

⚠️ Si le corpus ne contient pas l’information, je dois dire : **« Je ne sais pas. »**
"""


EXAMPLES = """🧪 **Exemples de messages à tester**

1) Notions
- *Définis une fraction et donne un exemple.*
- *Explique la simplification d’une fraction.*

2) Opérations
- *Comment additionner 1/2 et 3/4 ?*
- *Explique la multiplication de fractions avec un schéma mental.*

3) Erreurs fréquentes
- *Pourquoi 1/2 + 1/3 = 2/5 est faux ?*
- *Pourquoi certains élèves additionnent les dénominateurs ?*

4) Didactique
- *Rends didactique : “mettre au même dénominateur”.*

5) Classe
- */analyze*
- */export*
"""


def _split_user_message(content: str) -> List[str]:
    """Découpe un message en lignes non vides (gère les copier-coller multi-commandes)."""
    if not content:
        return []
    lines = [ln.strip() for ln in content.splitlines()]
    return [ln for ln in lines if ln]


def _parse_command(line: str) -> Tuple[str, str]:
    """
    Parse une commande de type:
      /analyze
      /analyze data/Students/responses.csv
      /analyze "C:\\path with spaces\\file.csv"
    Retourne: (cmd, arg)
    """
    # Exemple line = '/analyze "data/Students/my file.csv"'
    # On utilise shlex pour gérer les guillemets.
    tokens = shlex.split(line)
    cmd = tokens[0].lower() if tokens else ""
    arg = " ".join(tokens[1:]).strip() if len(tokens) > 1 else ""
    return cmd, arg


async def _handle_one_line(line: str) -> None:
    """Traite une ligne: commande ou question."""
    # Commandes (en minuscules)
    if line.lower() in {"/help", "help"}:
        await cl.Message(content=HELP).send()
        return

    if line.lower() in {"/examples", "examples"}:
        await cl.Message(content=EXAMPLES).send()
        return

    if line.lower() in {"/start", "start"}:
        await cl.Message(content=WELCOME).send()
        return

    if line.startswith("/"):
        cmd, arg = _parse_command(line)

        # /analyze [path]
        if cmd == "/analyze":
            payload = line if not arg else f"/analyze {arg}"
            answer = run_agent(payload)
            await cl.Message(content=answer).send()
            return

        # /export
        if cmd == "/export":
            answer = run_agent("/export")
            await cl.Message(content=answer).send()
            return

        # Commande inconnue
        await cl.Message(
            content=(
                "❓ Commande inconnue.\n"
                "Essaye `/help` pour voir les commandes disponibles."
            )
        ).send()
        return

    # Question “normale” (non commande)
    thinking = cl.Message(content="⏳ Je réfléchis…")
    await thinking.send()

    try:
        answer = run_agent(line)
        if not answer or not answer.strip():
            answer = "Désolé, je n’ai pas pu générer de réponse."
        thinking.content = answer
        await thinking.update()
    except Exception as e:
        thinking.content = (
            "⚠️ **Erreur interne** pendant le traitement.\n\n"
            f"**Détail :** `{type(e).__name__}`\n"
            "👉 Astuce : vérifie ton `.env` (clés), et que le corpus est bien présent.\n"
        )
        await thinking.update()
        raise


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    # ⭐️ Nouveau : support multi-lignes / multi-commandes
    lines = _split_user_message(message.content)

    if not lines:
        await cl.Message(content="Écris une question 🙂").send()
        return

    # On exécute chaque ligne dans l'ordre
    for line in lines:
        await _handle_one_line(line)
