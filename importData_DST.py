import requests
import pandas as pd
import json

# Test de connexion avec les clés API
api_key = 'e8hcgs9ady5xbaxd2b73n4zmq'
url_token = "https://api.lufthansa.com/v1/oauth/token"

API_KEY = 'kqUNaMx7yA'

headers = {
    "Content-Type": "application/x-www-form-urlencoded",
}
data = {
    "client_id": api_key,
    "client_secret": API_KEY,
    "grant_type": "client_credentials"
}

response = requests.post(url_token, headers=headers, data=data)

if response.status_code == 200:
    print("Connexion réussie !")
    access_token = response.json()["access_token"]
    print("Token d'accès :", access_token)
else:
    print("Erreur de connexion :", response.status_code)
    print(response.json())


# Structure des données

CLIENT_ID = 'e8hcgs9ady5xbaxd2b73n4zmq'
CLIENT_SECRET = 'kqUNaMx7yA'

api_key = 'e8hcgs9ady5xbaxd2b73n4zmq'
url_token = "https://api.lufthansa.com/v1/oauth/token"

headers_token = {
    "Content-Type": "application/x-www-form-urlencoded",
}
data_token = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "grant_type": "client_credentials"
}

response_token = requests.post(url_token, headers=headers_token, data=data_token)

# Effectuer la requête POST pour obtenir un token d'accès
response_token = requests.post(url_token, headers=headers_token, data=data_token)

# Vérifier le statut de la réponse
if response_token.status_code == 200:
    print("Connexion réussie !")
    access_token = response_token.json()["access_token"]
    print("Token d'accès :", access_token)
else:
    print("Erreur de connexion :", response_token.status_code)
    print(response_token.json())
    exit()

# Fonction pour obtenir les horaires des vols
def get_flight_schedules(access_token, origin, destination, date):
    url = f"https://api.lufthansa.com/v1/operations/schedules/{origin}/{destination}/{date}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Exemple d'utilisation
origin = 'FRA'  # Code IATA de l'aéroport de départ (Francfort)
destination = 'JFK'  # Code IATA de l'aéroport d'arrivée (New York JFK)
date = '2024-07-30' 

try:
    flight_data = get_flight_schedules(access_token, origin, destination, date)
    
    # Afficher les données brutes au format JSON
    print(json.dumps(flight_data, indent=4))
    
    # Extraire les informations pertinentes et les afficher
    if 'ScheduleResource' in flight_data and 'Schedule' in flight_data['ScheduleResource']:
        schedules = flight_data['ScheduleResource']['Schedule']
        print(f"Nombre de vols trouvés : {len(schedules)}")
        flights_info = []
        for schedule in schedules:
            total_journey = schedule.get('TotalJourney', {}).get('Duration', 'N/A')
            flight = schedule['Flight']
            if isinstance(flight, list):
                for item in flight:
                    flight_info = {
                        'FlightNumber': item['MarketingCarrier']['FlightNumber'],
                        'Departure': item['Departure']['ScheduledTimeLocal']['DateTime'],
                        'Arrival': item['Arrival']['ScheduledTimeLocal']['DateTime'],
                        'AircraftCode': item['Equipment']['AircraftCode'],
                        'TotalJourney': total_journey
                    }
                    flights_info.append(flight_info)
            elif isinstance(flight, dict):
                flight_info = {
                    'FlightNumber': flight['MarketingCarrier']['FlightNumber'],
                    'Departure': flight['Departure']['ScheduledTimeLocal']['DateTime'],
                    'Arrival': flight['Arrival']['ScheduledTimeLocal']['DateTime'],
                    'AircraftCode': flight['Equipment']['AircraftCode'],
                    'TotalJourney': total_journey
                }
                flights_info.append(flight_info)
            else:
                print("Le champ 'Flight' a un format inattendu :", flight)

        # Convertir en DataFrame
        df = pd.DataFrame(flights_info)
        

        # Convertir les colonnes de date/heure en datetime
        df['Departure'] = pd.to_datetime(df['Departure'])
        df['Arrival'] = pd.to_datetime(df['Arrival'])

        # Ajouter une colonne de durée de vol
        df['Duration'] = (df['Arrival'] - df['Departure']).dt.total_seconds() / 3600
        print(df)
    else:
        print("Aucune donnée de vol trouvée.")

except requests.exceptions.HTTPError as http_err:
    print(f"Erreur HTTP : {http_err}")
    print("Contenu de la réponse :", response_token.text)
except Exception as e:
    print(f"Erreur : {e}")
