import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
import json
from requests.exceptions import HTTPError

# Variables de configuration
API_KEY = 'e8hcgs9ady5xbaxd2b73n4zmq'
BASE_URL = 'https://api.lufthansa.com/v1'
MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority'  # Remplacez par l'URL de MongoDB Atlas
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights'



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

# Listes des aéroports 
airports = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'MIA', 'SFO', 'SEA', 'BOS', 'PHL','LHR', 'CDG', 'FRA', 'AMS', 'HKG', 'SYD', 'NRT', 'DXB', 'SIN', 'CAN']


# Fonction pour extraire les données depuis l'API Lufthansa
def fetch_flight_data(airport,date):
    endpoint = f'{BASE_URL}/operations/flightstatus/departures/{airport}/{date}'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    try:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()  # Lève une erreur pour les codes de statut 4xx/5xx
        return response.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred for {airport} : {http_err}')
    except Exception as err:
        print(f'Other error occurred for {airport}: {err}')
    return None  # Retourne None en cas d'erreur

# Fonction pour transformer les données
def transform_data(data):

    if 'FlightStatusResource' in data and 'Flights' in data['FlightStatusResource']:
        flights = data['FlightStatusResource']['Flights']
        flights_info = []
        flight = flights['Flight']
        if isinstance(flight, list):
            for item in flight:
                flight_info = {
                    'FlightNumber': item['MarketingCarrier']['FlightNumber'],
                    'Departure': item['Departure'],
                    'Arrival': item['Arrival'],
                    'FlightStatus' : item['FlightStatus'],
                    'Aircraft': item['Equipment']
                    }
                flights_info.append(flight_info)
        elif isinstance(flight, dict):
            flight_info = {
                'FlightNumber': item['MarketingCarrier']['FlightNumber'],
                'Departure': item['Departure'],
                'Arrival': item['Arrival'],
                'FlightStatus' : item['FlightStatus'],
                'Aircraft': item['Equipment']
            }
            flights_info.append(flight_info)
        else:
            print("Le champ 'Flight' a un format inattendu :", flight)
    return flights_info

# Fonction pour insérer les données dans MongoDB
def insert_data(data):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index([('Departure.ScheduledTimeUTC.DateTime', 1)], expireAfterSeconds=3600)
    #records = df.to_dict('records')
    #print(records)
    collection.insert_many(data)
    print("Insertion réussie")

# Script principal
 
def process_data(data):
    # Exemple de traitement des données, à adapter en fonction de votre besoin
    if data:
        print(json.dumps(data, indent=4))
        #df = pd.json_normalize(data)
        #print(df.head())  # Affiche les premiers résultats pour vérification
    else:
        print("No data received.")

if __name__ == '__main__':
    date = '2024-08-01T08:00' 
    for airport in airports:
        print(f'Retrieving data for {airport} at {date}')
        data = fetch_flight_data(airport,date)
        if data != None :
            flights_info = transform_data(data)
            insert_data(flights_info)

