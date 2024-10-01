# Utilisez l'image officielle Python
FROM python:3.11-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier requirements.txt pour un meilleur cache
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'application dans le conteneur
COPY . .

# Exposer le port sur lequel l'application FastAPI sera exécutée
EXPOSE 8000

# Commande pour exécuter l'application FastAPI avec uvicorn
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000", "--reload"]
