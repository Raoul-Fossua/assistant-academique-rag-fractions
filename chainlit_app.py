from __future__ import annotations

import chainlit as cl

# IMPORTANT :
# agents.py doit exposer une fonction :
#   run_agent(message: str) -> str
from agents import run_agent


WELCOME = """👋 Bonjour ! **Assistant pédagogique intelligent – Fractions (5e)**

Je peux :
- 📘 Expliquer une notion / une méthode sur les fractions (RAG + sources)
- 🧠 Expliquer une erreur fréquente (ex: *1/2 + 1/3 = 2/5*)
- ✍️ Reformuler une explication en version didactique
- 🧩 Préparer des profils / groupes de besoin à partir de `responses.csv`

Commandes :
- `/help` : affiche l’aide
- `/examples` : quelques idées de questions
"""


HELP = """📌 **Aide rapide**

✅ Exemples :
- *Résume les opérations sur les fractions.*
- *Explique “mettre au même dénominateur” avec du sens.*
- *Pourquoi certains élèves font 1/2 + 1/3 = 2/5 ?*
- *Donne une explication didactique de : “on multiplie en croix”.*
- *Analyse les profils d’erreurs de ma classe (responses.csv).*

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
- *Rends didactique : “on met au même dénominateur”.*

5) Classe
- *Fais la pré-analyse des profils à partir de responses.csv.*
"""


def _is_command(text: str) -> str | None:
    t = (text or "").strip().lower()
    if t in {"/help", "help"}:
        return "help"
    if t in {"/examples", "examples"}:
        return "examples"
    if t in {"/start", "start"}:
        return "start"
    return None


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_text = (message.content or "").strip()

    # ── Commandes ──────────────────────────────────────────────
    cmd = _is_command(user_text)
    if cmd == "help":
        await cl.Message(content=HELP).send()
        return
    if cmd == "examples":
        await cl.Message(content=EXAMPLES).send()
        return
    if cmd == "start":
        await cl.Message(content=WELCOME).send()
        return

    # ── Traitement normal ─────────────────────────────────────
    # Petit "thinking" UX
    msg = cl.Message(content="⏳ Je réfléchis…")
    await msg.send()

    try:
        # run_agent est synchrone → on l’appelle tel quel
        answer = run_agent(user_text)

        if not answer or not answer.strip():
            answer = "Désolé, je n’ai pas pu générer de réponse."

        msg.content = answer
        await msg.update()

    except Exception as e:
        # Erreur propre, sans crasher l’app
        msg.content = (
            "⚠️ **Erreur interne** pendant le traitement.\n\n"
            f"**Détail :** `{type(e).__name__}`\n"
            "👉 Astuce : vérifie ton `.env` (clés), et que le corpus est bien présent.\n"
        )
        await msg.update()
        # Pour debug console
        raise
