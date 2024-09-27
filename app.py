import streamlit as st
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt

# Connexion à MongoDB
MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority' 
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

CLIENT_MONGO = MongoClient(MONGO_URI)
DB = CLIENT_MONGO[DB_NAME]
COLLECTION = DB[COLLECTION_NAME]
# Fonction pour récupérer les données
def load_data():
    data = COLLECTION.find({})
    flights = []
    for flight in data:
        flight_data = {
            'FlightNumber': flight.get('FlightNumber'),
            'DepartureAirport': flight.get('DepartureAirport'),
            'ArrivalAirport': flight.get('ArrivalAirport'),
            'DepartureTimeLocal': flight.get('DepartureTimeLocal'),
            'ArrivalTimeLocal': flight.get('ArrivalTimeLocal'),
            'ArrivalDelayDuration': flight.get('ArrivalDelayDuration'),
            'DepartureTempC': flight.get('DepartureWeather', {}).get('temp_c', None),
            'DepartureHumidity': flight.get('DepartureWeather', {}).get('humidity', None),
            'DeparturePrecipMM': flight.get('DepartureWeather', {}).get('precip_mm', None),
            'DepartureWindKPH': flight.get('DepartureWeather', {}).get('wind_kph', None),
            'DepartureVisKM': flight.get('DepartureWeather', {}).get('vis_km', None),
            'DepartureGustKPH': flight.get('DepartureWeather', {}).get('gust_kph', None),
            'DepartureCondition': flight.get('DepartureWeather', {}).get('condition_text', None),
            'ArrivalTempC': flight.get('ArrivalWeather', {}).get('temp_c', None),
            'ArrivalHumidity': flight.get('ArrivalWeather', {}).get('humidity', None),
            'ArrivalPrecipMM': flight.get('ArrivalWeather', {}).get('precip_mm', None),
            'ArrivalWindKPH': flight.get('ArrivalWeather', {}).get('wind_kph', None),
            'ArrivalVisKM': flight.get('ArrivalWeather', {}).get('vis_km', None),
            'ArrivalGustKPH': flight.get('ArrivalWeather', {}).get('gust_kph', None),
            'ArrivalCondition': flight.get('ArrivalWeather', {}).get('condition_text', None)
        }
        flights.append(flight_data)
    return pd.DataFrame(flights)

# Titre de l'application
st.title("Analyse des Données de Vols")

# Chargement des données
flights_df = load_data()

# Afficher les données sous forme de tableau
if st.checkbox("Afficher les données brutes"):
    st.subheader("Données des vols")
    st.write(flights_df)

# Statistiques descriptives
st.subheader("Statistiques descriptives")
st.write(flights_df.describe())

# Visualisation des retards d'arrivée
st.subheader("Visualisation des Retards d'Arrivée")
plt.figure(figsize=(10, 5))
plt.hist(flights_df['ArrivalDelayDuration'], bins=30, color='blue', alpha=0.7)
plt.title("Distribution des Retards d'Arrivée")
plt.xlabel("Retard d'Arrivée (secondes)")
plt.ylabel("Nombre de Vols")
st.pyplot(plt)

# Visualisation des retards moyens par aéroport de départ
st.subheader("Retard Moyen par Aéroport de Départ")
average_delay = flights_df.groupby('DepartureAirport')['ArrivalDelayDuration'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.barh(average_delay['DepartureAirport'], average_delay['ArrivalDelayDuration'], color='orange')
plt.title("Retard Moyen par Aéroport de Départ")
plt.xlabel("Retard Moyen (secondes)")
plt.ylabel("Aéroport de Départ")
st.pyplot(plt)

# Visualisation des retards selon les conditions météorologiques
st.subheader("Retard par Conditions Météorologiques de Départ")
weather_delay = flights_df.groupby('DepartureCondition')['ArrivalDelayDuration'].mean().reset_index()
plt.figure(figsize=(10, 5))
plt.bar(weather_delay['DepartureCondition'], weather_delay['ArrivalDelayDuration'], color='green')
plt.xticks(rotation=45)
plt.title("Retard Moyen par Conditions Météorologiques de Départ")
plt.xlabel("Conditions Météorologiques")
plt.ylabel("Retard Moyen (secondes)")
st.pyplot(plt)

# Filtrage des données par aéroport de départ
departure_airport = st.selectbox("Sélectionnez un aéroport de départ", flights_df['DepartureAirport'].unique())
filtered_data = flights_df[flights_df['DepartureAirport'] == departure_airport]

if st.checkbox("Afficher les données filtrées"):
    st.subheader(f"Données pour l'Aéroport de Départ: {departure_airport}")
    st.write(filtered_data)

# Afficher les retards pour l'aéroport sélectionné
st.subheader(f"Retards d'Arrivée pour l'Aéroport de Départ: {departure_airport}")
st.write(filtered_data[['FlightNumber', 'ArrivalDelayDuration']])

# Exécuter l'application Streamlit
if __name__ == "__main__":
    st.write("L'application est prête à être exécutée.")
