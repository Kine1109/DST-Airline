import requests
import json

# Remplacez 'your_api_key_here' par votre clé API OpenWeatherMap
api_key = '7c922b7cdf8a0758d88b6899ffa0121c'
city_name = 'Paris'
base_url = "http://api.openweathermap.org/data/2.5/weather?"

# Construire l'URL complète
complete_url = f"{base_url}q={city_name}&appid={api_key}"

# Faire la requête à l'API
response = requests.get(complete_url)

# Vérifier si la requête a réussi
if response.status_code == 200:
    # Convertir les données de la réponse en JSON
    data = response.json()
    # Extraire les informations météorologiques
    main = data['main']
    weather = data['weather'][0]
    print(f"Temperature: {main['temp']}°K")
    print(f"Weather: {weather['description']}")

    # afficher tout le document
    print(json.dumps(data, indent=4))
else:
    # Gérer les erreurs, comme une clé API invalide
    print(f"Erreur: {response.status_code} - {response.json()['message']}")
