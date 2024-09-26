import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import mlflow.pyfunc
from pymongo import MongoClient


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

def load_model(run_id):
    """
    Chargement du modèle MLflow à partir d'un run ID spécifique.
    """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def load_preprocessors(run_id):
    """Charge les préprocesseurs à partir de MLflow."""
    #artifact_uri = f"runs:/{run_id}/preprocessors"
    artifact_uri = f"mlruns/0/{run_id}/artifacts/preprocessors"
    # Téléchargement des préprocesseurs à partir des artefacts MLflow
    #encoder_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/encoder.pkl")
    #scaler_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/scaler.pkl")
    #encoder_path = mlflow.get_artifact_uri("encoder.pkl")
    #scaler_path = mlflow.get_artifact_uri("scaler.pkl")
    # Chargement des préprocesseurs
    encoder_path = f"{artifact_uri}/encoder.pkl"
    scaler_path = f"{artifact_uri}/scaler.pkl"
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    return encoder, scaler

def preprocess_new_data(new_data, encoder, scaler):
    """
    Prétraitement des nouvelles données en utilisant les préprocesseurs chargés.
    """

    categorical_features = ['FlightNumber', 'DepartureAirport', 'ArrivalAirport', 'DepartureCondition', 'ArrivalCondition']
    continuous_features = ['DepartureTempC', 'DepartureHumidity', 'DeparturePrecipMM', 
                           'DepartureWindKPH', 'DepartureVisKM', 'DepartureGustKPH',
                           'ArrivalTempC', 'ArrivalHumidity', 'ArrivalPrecipMM', 
                           'ArrivalWindKPH', 'ArrivalVisKM', 'ArrivalGustKPH',
                           'DepartureHour', 'ArrivalHour', 'DepartureDayOfWeek', 'ArrivalDayOfWeek', 
                           'DepartureMonth', 'ArrivalMonth']
    
    # Encodage des variables catégorielles
    encoded_categorical = encoder.transform(new_data[categorical_features])
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_(categorical_features))

    # Normalisation les caractéristiques continues
    scaled_continuous = scaler.transform(new_data[continuous_features])
    scaled_df = pd.DataFrame(scaled_continuous, columns=continuous_features)

    # Concaténation des caractéristiques encodées et normalisées
    processed_data = pd.concat([scaled_df, encoded_df], axis=1)

    return processed_data

def main():
    # Spécifiez le run_id du modèle et des préprocesseurs
    run_id = "4d5c1a0716f9460bbc15d80fb3f8612a"
    
    # Chargement du modèle
    model = load_model(run_id)
    
    # Chargement des préprocesseurs
    encoder, scaler = load_preprocessors(run_id)


    # Configuration
    MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority' 
    db_name = 'flight_data'
    collection_name = 'flights_with_weather'

    # Connexion à la base de données
    collection = connect_to_mongodb(MONGO_URI, db_name, collection_name)
    # Récupérer le premier document de la collection
    first_document = collection.find_one()
    print(first_document)
    # Exemple de nouvelles données
    new_data = pd.DataFrame({
        'FlightNumber': ['012'],
        'DepartureAirport': ['JFK'],
        'ArrivalAirport': ['IST'],
        'DepartureCondition': ['Clear'],
        'ArrivalCondition': ['Sunny'],
        'DepartureTempC': [21.6],
        'DepartureHumidity': [54],
        'DeparturePrecipMM': [0.0],
        'DepartureWindKPH': [15.5],
        'DepartureVisKM': [10.0],
        'DepartureGustKPH': [21.8],
        'ArrivalTempC': [28.3],
        'ArrivalHumidity': [59],
        'ArrivalPrecipMM': [0.0],
        'ArrivalWindKPH': [25.2],
        'ArrivalVisKM': [10.0],
        'ArrivalGustKPH': [33.3],
        'DepartureHour': [0],
        'ArrivalHour': [17],
        'DepartureDayOfWeek': [5],  
        'ArrivalDayOfWeek': [5],    
        'DepartureMonth': [8],      
        'ArrivalMonth': [8],
         'De': [8]         
    })

    # Prétraitement des nouvelles données
    preprocessed_data = preprocess_new_data(new_data, encoder, scaler)
    
    # Vérification des colonnes
    print("Colonnes des données prétraitées:", preprocessed_data.columns.tolist())

    # Prédiction avec le modèle
    predictions = model.predict(preprocessed_data)
    predictions = np.maximum(predictions, 0)  # Ne permet pas de prédictions négatives
    
    # Affichage des prédictions
    print(f"Prédiction du retard d'arrivée (en minutes): {predictions}")

if __name__ == "__main__":
    main()
