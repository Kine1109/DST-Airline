import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import joblib

def connect_to_mongodb(mongo_uri, db_name, collection_name):
    """
    Se connecter à MongoDB Atlas et retourner la collection.

    Paramètres :
        mongo_uri (str) : L'URI MongoDB pour la connexion.
        db_name (str) : Le nom de la base de données.
        collection_name (str) : Le nom de la collection.

    Retourne :
        collection : L'objet collection de MongoDB.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return db[collection_name]

def extract_flight_data(collection):
    """
    Extrait les données de vol de la collection MongoDB et les convertit en DataFrame.

    Paramètres :
        collection : L'objet collection de MongoDB.

    Retourne :
        pd.DataFrame : Un DataFrame contenant les données de vol.
    """
    flights = collection.find({})
    data = []
    for flight in flights:
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
        data.append(flight_data)
    return pd.DataFrame(data)

def prepare_data(flights_df):
    """
    Prépare les données de vol en extrayant les caractéristiques, encodant les variables catégorielles 
    et normalisant les variables continues.

    Paramètres :
        flights_df (pd.DataFrame) : Le DataFrame contenant les données de vol.

    Retourne :
        X (pd.DataFrame) : Caractéristiques prétraitées pour l'entraînement du modèle.
        y (pd.Series) : Variable cible (retard à l'arrivée) pour l'entraînement du modèle.
        encoder (OneHotEncoder) : Encodeur utilisé pour transformer les variables catégorielles.
        scaler (MinMaxScaler) : Scaler utilisé pour normaliser les variables continues.
    """
    # Extraire l'heure, le jour de la semaine, et le mois des dates locales
    flights_df['DepartureHour'] = pd.to_datetime(flights_df['DepartureTimeLocal']).dt.hour
    flights_df['ArrivalHour'] = pd.to_datetime(flights_df['ArrivalTimeLocal']).dt.hour
    flights_df['DepartureDayOfWeek'] = pd.to_datetime(flights_df['DepartureTimeLocal']).dt.dayofweek
    flights_df['ArrivalDayOfWeek'] = pd.to_datetime(flights_df['ArrivalTimeLocal']).dt.dayofweek
    flights_df['DepartureMonth'] = pd.to_datetime(flights_df['DepartureTimeLocal']).dt.month
    flights_df['ArrivalMonth'] = pd.to_datetime(flights_df['ArrivalTimeLocal']).dt.month

    # Suppression des colonnes non nécessaires
    flights_df = flights_df.drop(columns=['DepartureTimeLocal', 'ArrivalTimeLocal'])

    # Encodage des variables catégorielles
    categorical_features = ['FlightNumber', 'DepartureAirport', 'ArrivalAirport', 'DepartureCondition', 'ArrivalCondition']
   
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    #encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_categorical = encoder.fit_transform(flights_df[categorical_features])

    # Normalisation des caractéristiques continues
    continuous_features = ['DepartureTempC', 'DepartureHumidity', 'DeparturePrecipMM', 
                           'DepartureWindKPH', 'DepartureVisKM', 'DepartureGustKPH',
                           'ArrivalTempC', 'ArrivalHumidity', 'ArrivalPrecipMM', 
                           'ArrivalWindKPH', 'ArrivalVisKM', 'ArrivalGustKPH',
                           'DepartureHour', 'ArrivalHour', 'DepartureDayOfWeek', 
                           'ArrivalDayOfWeek', 'DepartureMonth', 'ArrivalMonth']

    scaler = MinMaxScaler()
    flights_df[continuous_features] = scaler.fit_transform(flights_df[continuous_features])

    # Création du DataFrame final
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
    flights_df = pd.concat([flights_df.drop(columns=categorical_features), encoded_df], axis=1)

    # Sélection des caractéristiques (features) et de la cible (target)
    X = flights_df.drop(columns=['ArrivalDelayDuration'])
    y = flights_df['ArrivalDelayDuration']
    
    return X, y,encoder,scaler

def train_model(X_train, y_train):
    """
    Entraîne un modèle de régression linéaire sur les données d'entraînement.

    Paramètres :
        X_train (pd.DataFrame) : Caractéristiques pour l'entraînement.
        y_train (pd.Series) : Variable cible pour l'entraînement.

    Retourne :
        model : Le modèle de régression linéaire entraîné.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle entraîné en utilisant les données de test et retourne les métriques de performance.

    Paramètres :
        model : Le modèle entraîné.
        X_test (pd.DataFrame) : Caractéristiques pour le test.
        y_test (pd.Series) : Valeurs cibles réelles pour le test.

    Retourne :
        dict : Un dictionnaire contenant les métriques d'évaluation.
    """
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Ne permet pas de prédictions négatives

    # Convertir les retards en heures
    y_test = y_test / 60
    y_pred = y_pred / 60

    # Calcul des métriques de performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
        'r2_score': r2,
        'y_true': y_test,
        'y_pred': y_pred
    }

def log_experiment(metrics, model,encoder,scaler):
    """
    Enregistre les détails de l'expérience et le modèle à l'aide de MLflow.

    Paramètres :
        metrics (dict) : Les métriques à enregistrer.
        model : Le modèle entraîné à enregistrer.
        encoder (OneHotEncoder) : Encodeur utilisé pour transformer les variables catégorielles.
        scaler (MinMaxScaler) : Scaler utilisé pour normaliser les variables continues.
    """
    mlflow.start_run()

    # Sauvegarder les préprocesseurs dans le dossier 'artifacts'
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Loguer les préprocesseurs comme artefacts dans MLflow
    #mlflow.log_artifact('encoder.pkl', 'preprocessors')
    #mlflow.log_artifact('scaler.pkl', 'preprocessors')
    
    # Sauvegarder le modèle et les métriques
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mean_squared_error", metrics['mean_squared_error'])
    mlflow.log_metric("mean_absolute_error", metrics['mean_absolute_error'])
    mlflow.log_metric("r2_score", metrics['r2_score'])
    mlflow.sklearn.log_model(model, "model")
    mlflow.end_run()

def main():
    # Configuration
    MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority' 
    db_name = 'flight_data'
    collection_name = 'flights_with_weather'

    # Connexion à la base de données
    collection = connect_to_mongodb(MONGO_URI, db_name, collection_name)

    # Extraction des données
    flights_df = extract_flight_data(collection)

    # Préparation des données
    X, y,encoder,scaler = prepare_data(flights_df)

    # Diviser en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraînement du modèle
    model = train_model(X_train, y_train)

    # Évaluation du modèle
    metrics = evaluate_model(model, X_test, y_test)

    # Journalisation des résultats dans MLflow
    log_experiment(metrics, model,encoder,scaler)

    # Afficher les résultats
    for i in range(5):
        print(f"Vrai retard: {metrics['y_true'].values[i]}, Prédiction: {metrics['y_pred'][i]}")

if __name__ == "__main__":
    main()
