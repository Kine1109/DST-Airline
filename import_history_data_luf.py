import requests
import pandas as pd
from pymongo import MongoClient
from dateutil import parser
import logging
import json
from datetime import datetime, timedelta


# Configurer le module logging
logging.basicConfig(
    filename='app.log',  # Nom du fichier de log
    level=logging.ERROR,  # Niveau de log (ERROR pour capturer les erreurs uniquement)
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Variables de configuration
API_KEY = '4f2f845653e64a21b22163313242808'
BASE_URL = 'https://api.lufthansa.com/v1'
MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority' 
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

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
def read_airports_from_csv(file_path):
    """
    Lit les codes IATA des aéroports depuis un fichier CSV, en supprimant les lignes avec des codes IATA vides.
    
    Args:
        file_path (str): Chemin vers le fichier CSV contenant les codes IATA.
    
    Retourne:
        list: Liste des codes IATA des aéroports, sans codes vides.
    """
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['IATA'])  # Supprimer les lignes où la colonne IATA est NaN
    df = df[df['IATA'].astype(str).str.strip() != '']  # Supprimer les lignes où la colonne IATA est vide ou contient uniquement des espaces
    return df

#airports_df = read_airports_from_csv('airports_cleaned.csv')
#airports = airports_df['IATA'].tolist()
#print(len(airports))
airports = ['JFK', 'LAX', 'ORD', 'ATL', 'DFW', 'MIA', 'SFO', 'SEA', 'BOS', 'PHL','LHR', 'CDG', 'FRA', 'AMS', 
            'HKG', 'SYD', 'NRT', 'DXB', 'SIN', 'CAN','KEF','ALG','TUN','FMO','CGN','NUE','HAJ','PAD','STR',
            'HEL','CPH','CMN','ADD','TIA','TXL','MUC','ZRH','VIE','BER','HAM','DUS','MXP','BRU','MAD','BCN',
            'BKK','FCO','GVA','LIS','MAN','']
# Fonction pour extraire les données depuis l'API Lufthansa

def fetch_flight_data(airport, date):
    """
    Récupère les données de vol pour un aéroport donné et une date spécifique depuis l'API Lufthansa.

    Args:
        airport (str): Code IATA de l'aéroport.
        date (str): Date pour laquelle récupérer les données (format 'YYYY-MM-DD').

    Retourne:
        dict: Les données de vol en format JSON.
    """
    endpoint = f'{BASE_URL}/operations/flightstatus/departures/{airport}/{date}'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    print(endpoint)
    try:
        response = requests.get(endpoint, headers=headers)
        response.raise_for_status()  # Lève une erreur pour les codes de statut 4xx/5xx
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f'HTTP error occurred for {airport} : {http_err}')
    except Exception as err:
        logging.error(f'Other error occurred for {airport}: {err}')
    return None  # Retourne None en cas d'erreur

# Fonction pour transformer les données


def transform_data(data):
    """
    Transforme les données JSON de l'API Lufthansa en un format structuré.

    Args:
        data (dict): Les données de vol en format JSON.

    Retourne:
        list: Liste de dictionnaires contenant les informations des vols transformées.
    """
    if 'FlightStatusResource' in data and 'Flights' in data['FlightStatusResource']:
        flights = data['FlightStatusResource']['Flights']
        flights_info = []
        flight = flights['Flight']
        if isinstance(flight, list):
            for item in flight:
                DepartureTimeLocal = item['Departure']['ActualTimeLocal']['DateTime'] if 'ActualTimeLocal' in item['Departure'] else item['Departure']['ScheduledTimeLocal']['DateTime']
                DepartureDelayDuration = (parser.isoparse(item['Departure']['ScheduledTimeLocal']['DateTime']) - parser.isoparse(DepartureTimeLocal)).total_seconds() if 'ActualTimeLocal' in item['Departure'] else 0
                ArrivalTimeLocal = item['Arrival']['ActualTimeLocal']['DateTime'] if 'ActualTimeLocal' in item['Arrival'] else item['Arrival']['ScheduledTimeLocal']['DateTime']
                ArrivalDelayDuration = (parser.isoparse(item['Arrival']['ScheduledTimeLocal']['DateTime']) - parser.isoparse(DepartureTimeLocal)).total_seconds() if 'ActualTimeLocal' in item['Arrival'] else 0
                DepartureTimeUTC = item['Departure']['ActualTimeUTC']['DateTime'] if 'ActualTimeUTC' in item['Departure'] else item['Departure']['ScheduledTimeUTC']['DateTime']
                ArrivalTimeUTC = item['Arrival']['ActualTimeUTC']['DateTime'] if 'ActualTimeUTC' in item['Arrival'] else item['Arrival']['ScheduledTimeUTC']['DateTime']
                flight_info = {
                    'FlightNumber': item['MarketingCarrier']['FlightNumber'],
                    'DepartureAirport': item['Departure']['AirportCode'],
                    'DepartureTimeLocal': DepartureTimeLocal,
                    'DepartureTimeUTC': DepartureTimeUTC,
                    'DepartureDelayDuration': DepartureDelayDuration,
                    'ArrivalAirport': item['Arrival']['AirportCode'],
                    'ArrivalTimeLocal': ArrivalTimeLocal,
                    'ArrivalTimeUTC': ArrivalTimeUTC,
                    'ArrivalDelayDuration': ArrivalDelayDuration,
                    'AircraftCode': item['Equipment']['AircraftCode'],
                    'source':'lufthansa'
                }
                flights_info.append(flight_info)
        elif isinstance(flight, dict):
            item = flight
            DepartureTimeLocal = item['Departure']['ActualTimeLocal']['DateTime'] if 'ActualTimeLocal' in item['Departure'] else item['Departure']['ScheduledTimeLocal']['DateTime']
            DepartureDelayDuration = (parser.isoparse(item['Departure']['ScheduledTimeLocal']['DateTime']) - parser.isoparse(DepartureTimeLocal)).total_seconds() if 'ActualTimeLocal' in item['Departure'] else 0
            ArrivalTimeLocal = item['Arrival']['ActualTimeLocal']['DateTime'] if 'ActualTimeLocal' in item['Arrival'] else item['Arrival']['ScheduledTimeLocal']['DateTime']
            ArrivalDelayDuration = (parser.isoparse(item['Arrival']['ScheduledTimeLocal']['DateTime']) - parser.isoparse(DepartureTimeLocal)).total_seconds() if 'ActualTimeLocal' in item['Arrival'] else 0
            DepartureTimeUTC = item['Departure']['ActualTimeUTC']['DateTime'] if 'ActualTimeUTC' in item['Departure'] else item['Departure']['ScheduledTimeUTC']['DateTime']
            ArrivalTimeUTC = item['Arrival']['ActualTimeUTC']['DateTime'] if 'ActualTimeUTC' in item['Arrival'] else item['Arrival']['ScheduledTimeUTC']['DateTime']
            flight_info = {
                'FlightNumber': item['MarketingCarrier']['FlightNumber'],
                'DepartureAirport': item['Departure']['AirportCode'],
                'DepartureTimeLocal': DepartureTimeLocal,
                'DepartureTimeUTC': DepartureTimeUTC,
                'DepartureDelayDuration': DepartureDelayDuration,
                'ArrivalAirport': item['Arrival']['AirportCode'],
                'ArrivalTimeLocal': ArrivalTimeLocal,
                'ArrivalTimeUTC': ArrivalTimeUTC,
                'ArrivalDelayDuration': ArrivalDelayDuration,
                'AircraftCode': item['Equipment']['AircraftCode'],
                'source':'lufthansa'
            }
            flights_info.append(flight_info)
        else:
            print("Le champ 'Flight' a un format inattendu :", flight)
    return flights_info

# Fonction pour insérer les données dans MongoDB
def insert_data(data):
    """
    Insère les données dans une collection MongoDB.

    Args:
        data (list): Liste de dictionnaires contenant les informations des vols.
    """
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.create_index([('DepartureTimeUTC', 1)], expireAfterSeconds=3600)
    collection.insert_many(data)
    print("Insertion réussie")

# Pour les données météos
def get_weather_data(api_key, iata_code, date):
    """
    Récupère les données météorologiques historiques pour un aéroport donné depuis l'API WeatherAPI.

    Args:
        api_key (str): Clé API pour WeatherAPI.
        iata_code (str): Code IATA de l'aéroport.
        date (str): Date pour laquelle récupérer les données météo (format 'YYYY-MM-DD').

    Retourne:
        pd.DataFrame: DataFrame contenant les prévisions horaires météorologiques.
    """
    url = f'https://api.weatherapi.com/v1/history.json?key={api_key}&q={iata_code}&dt={date}'
    print(url)
    response = requests.get(url)
    data = response.json()
    
    location = data.get('location', {})
    forecast_hours = data.get('forecast', {}).get('forecastday', [])[0].get('hour', [])
    forecast_df = pd.DataFrame(forecast_hours)
    
    forecast_df['condition_text'] = [hour.get('condition', {}).get('text', 'N/A') for hour in forecast_hours]
    forecast_df['condition_code'] = [hour.get('condition', {}).get('code', 'N/A') for hour in forecast_hours]
    forecast_df['vis_km'] = [hour.get('vis_km', 'N/A') for hour in forecast_hours]
    forecast_df['gust_kph'] = [hour.get('gust_kph', 'N/A') for hour in forecast_hours]
    
    
    forecast_df = forecast_df[['time', 'temp_c', 'humidity', 'precip_mm', 'wind_kph', 'condition_text', 'condition_code', 'vis_km', 'gust_kph']]
    return forecast_df

def find_closest_weather_time(weather_df, target_time):
    """
    Trouve l'entrée météo la plus proche de l'heure cible.

    Args:
        weather_df (pd.DataFrame): DataFrame contenant les données météorologiques horaires.
        target_time (str): Heure cible pour trouver les données météo les plus proches.

    Retourne:
        pd.DataFrame: La ligne de données météo la plus proche de l'heure cible.
    """
    target_datetime = pd.to_datetime(target_time)
    weather_df['time'] = pd.to_datetime(weather_df['time'])
    closest_weather = weather_df.iloc[(weather_df['time'] - target_datetime).abs().argsort()[:1]]
    return closest_weather

# Script principal
def process_data(data):
    """
    Traite les données de vol en ajoutant les données météorologiques les plus proches et les insère dans MongoDB.

    Args:
        data (list): Liste de dictionnaires contenant les informations des vols.
    """
    if data:
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        for flight in data:
            departure_airport = flight['DepartureAirport']
            arrival_airport = flight['ArrivalAirport']
            date = flight['DepartureTimeUTC'][:10]  # Utiliser la date de départ pour les prévisions météorologiques
            
            # Récupérer les données météo pour les aéroports de départ et d'arrivée
            departure_weather_df = get_weather_data(API_KEY, departure_airport, date)
            arrival_weather_df = get_weather_data(API_KEY, arrival_airport, date)
            
            # Trouver les données météo les plus proches
            departure_weather = find_closest_weather_time(departure_weather_df, flight['DepartureTimeLocal'])
            arrival_weather = find_closest_weather_time(arrival_weather_df, flight['ArrivalTimeLocal'])
            
            # Ajouter les données météo au dictionnaire du vol
            flight['DepartureWeather'] = departure_weather.to_dict(orient='records')[0] if not departure_weather.empty else {}
            flight['ArrivalWeather'] = arrival_weather.to_dict(orient='records')[0] if not arrival_weather.empty else {}
        
        insert_data(data)
        print("Données sauvegardées dans MongoDB.")

if __name__ == '__main__':
    start_date_str = '2024-09-12T00:00'
    end_date_str = '2024-09-19T00:00'

    start_date = datetime.strptime(start_date_str, '%Y-%m-%dT%H:%M')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%dT%H:%M')
    interval = timedelta(days=1)  
    current_date = start_date
    while current_date <= end_date:
        all_flights_info = []
        formatted_date = current_date.strftime('%Y-%m-%dT%H:%M')
        for airport in airports:
            print(f'Retrieving data for {airport} on {formatted_date}')
            data = fetch_flight_data(airport, formatted_date)
            print(data)
            if data:
                flights_info = transform_data(data)
                all_flights_info.extend(flights_info)
    
        process_data(all_flights_info)
        current_date += interval
    
    
    
