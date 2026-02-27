# ==============================
# Dockerfile - HF Space
# Assistant pédagogique Fractions 5e
# ==============================

FROM python:3.11-slim

# 🔒 Créer un user non-root (recommandé HF)
RUN useradd -m -u 1000 user
USER user

# 🔧 PATH pour pip local
ENV PATH="/home/user/.local/bin:$PATH"

# 📁 Dossier de travail
WORKDIR /app

# Copier requirements d'abord (cache Docker)
COPY --chown=user requirements.txt requirements.txt

# Installer dépendances
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY --chown=user . /app

# ⚠️ IMPORTANT : Hugging Face impose le port 7860
EXPOSE 7860

# 🚀 Lancement Chainlit
CMD ["chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "7860"]



