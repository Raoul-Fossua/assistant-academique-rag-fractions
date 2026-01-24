@"
# Assistant Académique — RAG Fractions 5e (Chainlit + LangChain)

## Objectif
Assistant pédagogique spécialisé Fractions 5e : RAG sur corpus (PDF + Excel erreurs/remédiations) + outils (didactique, profils).

## Stack
- Chainlit
- LangChain + langchain-openai + langchain-community
- FAISS (vector store)
- OpenAI (LLM + embeddings)
- Tavily (recherche web)

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U -r requirements.txt
