import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from pymongo import MongoClient
from pathlib import Path



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
    Chargement du modèle à partir d'un run ID spécifique.
    """
    model_uri = f"{run_id}/model.pkl" #./mlruns/0{run_id}/model
    #model = mlflow.sklearn.load_model(model_uri)
    model = joblib.load(model_uri)
    return model

def load_preprocessors(run_id):
    """Charger les préprocesseurs."""
    
    encoder_path = f"{run_id}/encoder.pkl"
    scaler_path = f"{run_id}/scaler.pkl"
    encoder = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)
    
    return encoder, scaler

def preprocess_new_data(new_data, encoder, scaler):
    """
    Prétraitement des nouvelles données en utilisant les préprocesseurs chargés.
    """

    categorical_features = ['DepartureAirport', 'ArrivalAirport', 'DepartureCondition', 'ArrivalCondition']
    continuous_features = ['DepartureTempC', 'DepartureHumidity', 'DeparturePrecipMM', 
                           'DepartureWindKPH', 'DepartureVisKM', 'DepartureGustKPH',
                           'ArrivalTempC', 'ArrivalHumidity', 'ArrivalPrecipMM', 
                           'ArrivalWindKPH', 'ArrivalVisKM', 'ArrivalGustKPH',
                           'DepartureHour', 'ArrivalHour', 'DepartureDayOfWeek', 'ArrivalDayOfWeek', 
                           'DepartureMonth', 'ArrivalMonth']
    
    # Encodage des variables catégorielles
    encoded_categorical = encoder.transform(new_data[categorical_features])
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

    # Normalisation les caractéristiques continues
    scaled_continuous = scaler.transform(new_data[continuous_features])
    scaled_df = pd.DataFrame(scaled_continuous, columns=continuous_features)

    # Concaténation des caractéristiques encodées et normalisées
    processed_data = pd.concat([scaled_df, encoded_df], axis=1)

    return processed_data

def main(new_data):
    # Spécifiez le run_id du modèle et des préprocesseurs
    pathfile = Path(__file__).resolve().parent
    run_id = pathfile/"2"
    
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
    

    # Prétraitement des nouvelles données
    preprocessed_data = preprocess_new_data(new_data, encoder, scaler)
    
    # Prédiction avec le modèle
    predictions = model.predict(preprocessed_data)
    predictions = float(np.maximum(predictions, 0)) # Ne permet pas de prédictions négatives puis conversion en minutes
    
 
    # Affichage des prédictions
    return predictions

if __name__ == "__main__":
    main()
