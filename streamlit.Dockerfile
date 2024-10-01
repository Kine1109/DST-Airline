# Utiliser l'image officielle Python
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier requirements.txt d'abord pour un meilleur cache
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel Streamlit va s'exécuter
EXPOSE 8501

# Commande pour exécuter l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
