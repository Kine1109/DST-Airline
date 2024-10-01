import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
import requests
from datetime import datetime
import numpy as np
import os



# Connexion à MongoDB
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

CLIENT_MONGO = MongoClient(MONGO_URI)
DB = CLIENT_MONGO[DB_NAME]
COLLECTION = DB[COLLECTION_NAME]
API_KEY = os.getenv('API_KEY')


# Fonction pour récupérer les données depuis MongoDB
def load_data():
    data = COLLECTION.find({})
    flights = []
    for flight in data:
        flight_data = {
            'FlightNumber': flight.get('FlightNumber'),
            'DepartureAirport': flight.get('DepartureAirport'),  # Code IATA
            'ArrivalAirport': flight.get('ArrivalAirport'),      # Code IATA
            'DepartureTimeLocal': flight.get('DepartureTimeLocal'),
            'ArrivalTimeLocal': flight.get('ArrivalTimeLocal'),
            'ArrivalDelayDuration': flight.get('ArrivalDelayDuration'),
            'DepartureTempC': flight.get('DepartureWeather', {}).get('temp_c', None),
            'DepartureHumidity': flight.get('DepartureWeather', {}).get('humidity', None),
            'DeparturePrecipMM': flight.get('DepartureWeather', {}).get('precip_mm', None),
            'DepartureWindKPH': flight.get('DepartureWeather', {}).get('wind_kph', None),
            'DepartureCondition': flight.get('DepartureWeather', {}).get('condition_text', None),
            'ArrivalTempC':flight.get('ArrivalWeather', {}).get('temp_c', None),
            'ArrivalHumidity': flight.get('ArrivalWeather', {}).get('humidity', None),
            'ArrivalPrecipMM':flight.get('ArrivalWeather', {}).get('precip_mm', None),
            'ArrivalWindKPH': flight.get('ArrivalWeather', {}).get('wind_kph', None),
            'ArrivalCondition': flight.get('ArrivalWeather', {}).get('condition_text', None)
        }
        flights.append(flight_data)
    return pd.DataFrame(flights)

def get_weather_data(api_key, iata_code, date, target_time):
    """
    Récupère les données météorologiques historiques pour un aéroport donné à l'heure la plus proche de l'heure cible.
    
    Args:
        api_key (str): Clé API pour WeatherAPI.
        iata_code (str): Code IATA de l'aéroport.
        date (str): Date pour laquelle récupérer les données météo (format 'YYYY-MM-DD').
        target_time (datetime): Heure cible pour laquelle on veut récupérer les données météo.

    Retourne:
        pd.Series: La ligne contenant les prévisions horaires pour l'heure la plus proche de l'heure cible.
    """
    # Appeler l'API WeatherAPI pour récupérer les prévisions horaires pour la journée donnée
    url = f'https://api.weatherapi.com/v1/history.json?key={api_key}&q={iata_code}&dt={date}'
    response = requests.get(url)
    data = response.json()
    
    forecast_hours = data.get('forecast', {}).get('forecastday', [])[0].get('hour', [])
    forecast_df = pd.DataFrame(forecast_hours)
    forecast_df['condition_text'] = [hour.get('condition', {}).get('text', 'N/A') for hour in forecast_hours]
    forecast_df['condition_code'] = [hour.get('condition', {}).get('code', 'N/A') for hour in forecast_hours]
    
    # Convertir la colonne 'time' en datetime
    forecast_df['time'] = pd.to_datetime(forecast_df['time'])
    
    # Calculer la différence entre chaque heure et l'heure cible
    forecast_df['time_diff'] = abs(forecast_df['time'] - target_time)
    
    # Sélectionner la ligne avec la différence de temps la plus petite (l'heure la plus proche)
    closest_weather = forecast_df.loc[forecast_df['time_diff'].idxmin()]
    
    # Sélectionner les colonnes pertinentes
    closest_weather = closest_weather[['time', 'temp_c', 'humidity', 'precip_mm', 'wind_kph', 'condition_text', 'condition_code', 'vis_km', 'gust_kph']]
    
    return closest_weather

def custom_serializer(obj):
    if isinstance(obj, (np.int64, np.float64)):
        return obj.item()  # Convertir en type natif
    raise TypeError(f"Type {type(obj)} not serializable")

# Chargement du fichier des aéroports
@st.cache_data
def load_airport_data():
    return pd.read_csv('airports_cleaned.csv')

# Charger les données des vols et des aéroports
flights_df = load_data()
airports_df = load_airport_data()

# Convertir les secondes en minutes
flights_df['ArrivalDelayDuration'] = flights_df['ArrivalDelayDuration'] / 60

# Supprimer les lignes avec des codes IATA manquants
airports_df_cleaned = airports_df.dropna(subset=['IATA'])

# Créer un dictionnaire pour mapper les codes IATA aux noms complets des aéroports
iata_to_name = airports_df_cleaned.set_index('IATA')['Name'].to_dict()

# Titre de l'application
st.title("Analyse des Données de Vols")

# Ajout du menu de navigation à gauche
menu = st.sidebar.radio(
    "Menu de Navigation",
    ["Accueil", "Analyse des Retards", "Conditions Météorologiques", "Données Filtrées", "Faire une Prédiction"]
)

# Page Accueil
if menu == "Accueil":
    st.subheader("Bienvenue dans l'Analyse des Données de Vols")
    st.write("Cette application vous permet d'explorer les retards des vols ainsi que leur relation avec les conditions météorologiques.")
    
    # Afficher les statistiques descriptives globales
    st.subheader("Statistiques descriptives globales")
    st.write(flights_df.describe())

# Page Analyse des Retards
elif menu == "Analyse des Retards":
    st.subheader("Analyse des Retards d'Arrivée")
    # Retards moyens par aéroport d'arrivée
    average_delay = flights_df.groupby('ArrivalAirport')['ArrivalDelayDuration'].mean().reset_index()

    # Remplacer les codes IATA par les noms d'aéroports
    average_delay['ArrivalAirport'] = average_delay['ArrivalAirport'].map(iata_to_name)

    # Création du graphique interactif avec Plotly
    fig_delay = px.bar(
        average_delay, 
        x='ArrivalAirport', 
        y='ArrivalDelayDuration', 
        title="Retard Moyen par Aéroport d'Arrivée",
        labels={"ArrivalAirport": "Aéroport d'Arrivée", "ArrivalDelayDuration": "Retard Moyen (minutes)"}
    )
    st.plotly_chart(fig_delay, use_container_width=True)

    # Retards moyens par aéroport de départ
    average_delay = flights_df.groupby('DepartureAirport')['ArrivalDelayDuration'].mean().reset_index()

    # Remplacer les codes IATA par les noms d'aéroports
    average_delay['DepartureAirport'] = average_delay['DepartureAirport'].map(iata_to_name)

    # Création du graphique interactif avec Plotly
    fig_delay = px.bar(
        average_delay, 
        x='DepartureAirport', 
        y='ArrivalDelayDuration', 
        title="Retard Moyen par Aéroport de Départ",
        labels={"DepartureAirport": "Aéroport de Départ", "ArrivalDelayDuration": "Retard Moyen (minutes)"}
    )
    st.plotly_chart(fig_delay, use_container_width=True)

# Page Conditions Météorologiques
elif menu == "Conditions Météorologiques":
    st.subheader("Analyse des Retards par Conditions Météorologiques")

    # Liste déroulante pour choisir entre départ ou arrivée
    condition_choice = st.selectbox(
        "Choisissez les données météorologiques", 
        ["Départ", "Arrivée"]
    )

    # Choix des champs à analyser
    if condition_choice == "Départ":
        temp_col = 'DepartureTempC'
        humidity_col = 'DepartureHumidity'
        precip_col = 'DeparturePrecipMM'
        wind_col = 'DepartureWindKPH'
        condition_col = 'DepartureCondition'
    else:
        temp_col = 'ArrivalTempC'
        humidity_col = 'ArrivalHumidity'
        precip_col = 'ArrivalPrecipMM'
        wind_col = 'ArrivalWindKPH'
        condition_col = 'ArrivalCondition'

    # Retards moyens par conditions météorologiques
    st.subheader(f"Retard par Température ({condition_choice})")
    fig_temp = px.scatter(
        flights_df, x=temp_col, y='ArrivalDelayDuration',
        title=f"Retard par Température ({condition_choice})",
        labels={temp_col: "Température (°C)", "ArrivalDelayDuration": "Retard d'Arrivée (minutes)"}
    )
    st.plotly_chart(fig_temp, use_container_width=True)

    st.subheader(f"Retard par Humidité ({condition_choice})")
    fig_humidity = px.scatter(
        flights_df, x=humidity_col, y='ArrivalDelayDuration',
        title=f"Retard par Humidité ({condition_choice})",
        labels={humidity_col: "Humidité (%)", "ArrivalDelayDuration": "Retard d'Arrivée (minutes)"}
    )
    st.plotly_chart(fig_humidity, use_container_width=True)

    st.subheader(f"Retard par Précipitations ({condition_choice})")
    fig_precip = px.scatter(
        flights_df, x=precip_col, y='ArrivalDelayDuration',
        title=f"Retard par Précipitations ({condition_choice})",
        labels={precip_col: "Précipitations (mm)", "ArrivalDelayDuration": "Retard d'Arrivée (minutes)"}
    )
    st.plotly_chart(fig_precip, use_container_width=True)

    st.subheader(f"Retard par Vitesse du Vent ({condition_choice})")
    fig_wind = px.scatter(
        flights_df, x=wind_col, y='ArrivalDelayDuration',
        title=f"Retard par Vitesse du Vent ({condition_choice})",
        labels={wind_col: "Vitesse du Vent (KPH)", "ArrivalDelayDuration": "Retard d'Arrivée (minutes)"}
    )
    st.plotly_chart(fig_wind, use_container_width=True)

    st.subheader(f"Retard par Conditions Météorologiques ({condition_choice})")
    # Retards moyens par conditions météorologiques de départ
    weather_delay = flights_df.groupby(condition_col)['ArrivalDelayDuration'].mean().reset_index()
    # Création du graphique interactif avec Plotly
    fig_weather = px.bar(
        weather_delay, 
        x=condition_col, 
        y='ArrivalDelayDuration', 
        title=f"Retard par Conditions Météorologiques ({condition_choice})",
        labels={"DepartureCondition": "Conditions Météorologiques", "ArrivalDelayDuration": "Retard Moyen (minutes)"},
    )
    st.plotly_chart(fig_weather, use_container_width=True)

# Page Données Filtrées
elif menu == "Données Filtrées":
    st.subheader("Filtrer les Données par Aéroport de Départ")

    # Créer un ensemble de tous les codes IATA présents dans flights_df
    departure_airports = flights_df['DepartureAirport'].unique()
    #arrival_airports = flights_df['ArrivalAirport'].unique()

    # Filtrer airports_df_cleaned pour ne garder que les aéroports qui sont dans flights_df
    departure_airports_df_cleaned = airports_df_cleaned[airports_df_cleaned['IATA'].isin(departure_airports)]
    #arrival_airports_df_cleaned = airports_df_cleaned[airports_df_cleaned['IATA'].isin(arrival_airports)]

    # Sélection des informations de vol via des listes déroulantes avec les noms d'aéroports
    departure_airport_name = st.selectbox("Aéroport de départ", departure_airports_df_cleaned['Name'])
    #arrival_airport_name = st.selectbox("Aéroport d'arrivée", arrival_airports_df_cleaned['Name'])

    # Récupérer les codes IATA correspondants
    departure_airport_code = departure_airports_df_cleaned[departure_airports_df_cleaned['Name'] == departure_airport_name]['IATA'].values[0]
    #arrival_airport_code = arrival_airports_df_cleaned[arrival_airports_df_cleaned['Name'] == arrival_airport_name]['IATA'].values[0]
    
    
    # Filtrage des données par aéroport de départ
    #departure_airport_code = st.selectbox("Sélectionnez un aéroport de départ", flights_df['DepartureAirport'].unique())
    #departure_airport_name = iata_to_name.get(departure_airport_code, departure_airport_code)
    
    filtered_data = flights_df[flights_df['DepartureAirport'] == departure_airport_code]

    if st.checkbox("Afficher les données filtrées"):
        st.subheader(f"Données pour l'Aéroport de Départ: {departure_airport_name}")
        st.write(filtered_data)

    # Afficher les retards pour l'aéroport sélectionné
    st.subheader(f"Retards d'Arrivée pour l'Aéroport de Départ: {departure_airport_name}")
    st.write(filtered_data[['FlightNumber', 'ArrivalDelayDuration']])

# Page Faire une Prédiction
elif menu == "Faire une Prédiction":
    st.subheader("Faire une Prédiction de Retard")

    # Créer un ensemble de tous les codes IATA présents dans flights_df
    departure_airports = flights_df['DepartureAirport'].unique()
    arrival_airports = flights_df['ArrivalAirport'].unique()

    # Filtrer airports_df_cleaned pour ne garder que les aéroports qui sont dans flights_df
    departure_airports_df_cleaned = airports_df_cleaned[airports_df_cleaned['IATA'].isin(departure_airports)]
    arrival_airports_df_cleaned = airports_df_cleaned[airports_df_cleaned['IATA'].isin(arrival_airports)]

    # Sélection des informations de vol via des listes déroulantes avec les noms d'aéroports
    departure_airport_name = st.selectbox("Aéroport de départ", departure_airports_df_cleaned['Name'])
    arrival_airport_name = st.selectbox("Aéroport d'arrivée", arrival_airports_df_cleaned['Name'])

    # Récupérer les codes IATA correspondants
    departure_airport_code = departure_airports_df_cleaned[departure_airports_df_cleaned['Name'] == departure_airport_name]['IATA'].values[0]
    arrival_airport_code = arrival_airports_df_cleaned[arrival_airports_df_cleaned['Name'] == arrival_airport_name]['IATA'].values[0]
    
    # Sélection de la date et l'heure de départ
    departure_datetime = st.date_input("Sélectionnez la date de départ", datetime.now())
    departure_time = st.time_input("Sélectionnez l'heure de départ",key = 'departure_time')

    # Sélection de la date et l'heure d'arrivée
    arrival_datetime = st.date_input("Sélectionnez la date d'arrivée", datetime.now())
    arrival_time = st.time_input("Sélectionnez l'heure d'arrivée",key='arrival_time')

    # Prendre la date et l'heure de départ
    departure_full_datetime = datetime.combine(departure_datetime, departure_time)
    departure_date_str = departure_datetime.strftime('%Y-%m-%d')

    # Récupérer les données météo pour le départ à l'heure la plus proche
    departure_weather_closest = get_weather_data(API_KEY, departure_airport_code, departure_date_str, departure_full_datetime)

    # Prendre la date et l'heure d'arrivée
    arrival_full_datetime = datetime.combine(arrival_datetime, arrival_time)
    arrival_date_str = arrival_datetime.strftime('%Y-%m-%d')
    # Récupérer les données météo pour l'arrivée à l'heure la plus proche
    arrival_weather_closest = get_weather_data(API_KEY, arrival_airport_code, arrival_date_str, arrival_full_datetime)

    # Afficher les données météorologiques récupérées pour le départ et l'arrivée
    st.subheader("Conditions météorologiques pour le départ (heure la plus proche)")
    st.write(departure_weather_closest)

    st.subheader("Conditions météorologiques pour l'arrivée (heure la plus proche)")
    st.write(arrival_weather_closest)
    condition_text = arrival_weather_closest['condition_text']
    print(f"Conditions météo pour l'arrivée : {condition_text}")

    # Bouton pour lancer la prédiction
    if st.button("Prédire le Retard"):
        # Construction des données pour l'API de prédiction
        prediction_data = {
            
            'DepartureAirport': departure_airport_code,
            'ArrivalAirport': arrival_airport_code,
            'DepartureCondition': departure_weather_closest['condition_text'],
            'ArrivalCondition': arrival_weather_closest['condition_text'],
            'DepartureTempC':float(departure_weather_closest['temp_c']),
            'DepartureHumidity': int(departure_weather_closest['humidity']),
            'DeparturePrecipMM': float(departure_weather_closest['precip_mm']),
            'DepartureWindKPH': float(departure_weather_closest['wind_kph']),
            'DepartureVisKM': float(departure_weather_closest['vis_km']),
            'DepartureGustKPH': float(departure_weather_closest['gust_kph']),
            'ArrivalTempC': float(arrival_weather_closest['temp_c']),
            'ArrivalHumidity': int(arrival_weather_closest['humidity']),
            'ArrivalPrecipMM': float(arrival_weather_closest['precip_mm']),
            'ArrivalWindKPH': float(arrival_weather_closest['wind_kph']),
            'ArrivalVisKM': float(arrival_weather_closest['vis_km']),
            'ArrivalGustKPH': float(arrival_weather_closest['gust_kph']),
            'DepartureHour': departure_full_datetime.hour,
            'ArrivalHour' : arrival_full_datetime.hour,
            'DepartureDayOfWeek' : departure_full_datetime.weekday() + 1,
            'ArrivalDayOfWeek' : arrival_full_datetime.weekday() + 1,
            'DepartureMonth' : departure_full_datetime.month,
            'ArrivalMonth':arrival_full_datetime.month

        }

        # Appel à l'API de prédiction (URL fictive, à remplacer par votre propre API)
        url = "http://fastapi:8000/predict"  # Remplacez par l'URL de votre API de prédiction
        print(url)
        print(prediction_data)
        #prediction_data = json.dumps(prediction_data, default=custom_serializer)
        #print(prediction_data)
        response = requests.post(url, json=prediction_data)


        if response.status_code == 200:
            prediction_result = response.json().get('prediction_delay')
            # Calculer les heures, minutes et secondes
            hours = int(prediction_result // 3600)
            minutes = int((prediction_result % 3600) // 60)
            # # Formatage en 'h:mm:ss'
            result = f"Prédiction du retard d'arrivée: {round(prediction_result  / 60,2)} minutes soit {hours} heures {minutes:02} minutes"
            st.success(result)
        else:
            st.error("Erreur lors de la prédiction. Veuillez réessayer.")


