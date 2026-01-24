import chainlit as cl
from agents import agent

WELCOME = (
    "👋 Bonjour ! Assistant pédagogique intelligent – **Fractions**.\n\n"
    "Je peux :\n"
    "1) Expliquer une notion ou une erreur fréquente sur les fractions (RAG, avec sources)\n"
    "2) Reformuler une explication en version didactique\n"
    "3) Préparer des groupes de besoin à partir de `responses.csv`\n\n"
    "Commandes : `/help`"
)

HELP = (
    "🧭 Aide\n\n"
    "Exemples :\n"
    "- Pourquoi les élèves ajoutent les dénominateurs ?\n"
    "- Explique la différence entre fraction et quotient.\n"
    "- Donne une remédiation sur 'mettre au même dénominateur'.\n"
    "- Fais les groupes de besoin (à partir du CSV).\n"
)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", agent)
    await cl.Message(content=WELCOME).send()

@cl.on_message
async def on_message(message: cl.Message):
    txt = (message.content or "").strip()
    if not txt:
        return
    if txt.lower() == "/help":
        await cl.Message(content=HELP).send()
        return

    ag = cl.user_session.get("agent")
    res = await cl.make_async(ag.invoke)({"input": txt})
    answer = res.get("output", "Désolé, je n’ai pas pu générer de réponse.")
    await cl.Message(content=answer).send()
