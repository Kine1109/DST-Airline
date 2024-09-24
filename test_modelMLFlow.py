import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


def load_model(run_id):
    """
    Chargement du modèle MLflow à partir d'un run ID spécifique.
    """
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def load_preprocessors(run_id):
    """Charge les préprocesseurs à partir de MLflow."""
    artifact_uri = f"runs:/{run_id}/preprocessors"
    
    # Téléchargement des préprocesseurs à partir des artefacts MLflow
    #encoder_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/encoder.pkl")
    #scaler_path = mlflow.artifacts.download_artifacts(f"{artifact_uri}/scaler.pkl")
    #encoder_path = mlflow.get_artifact_uri("encoder.pkl")
    #scaler_path = mlflow.get_artifact_uri("scaler.pkl")
    # Chargement des préprocesseurs
    encoder = joblib.load("encoder.pkl")
    scaler = joblib.load("scaler.pkl")
    
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
    encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names(categorical_features))

    # Normalisation les caractéristiques continues
    scaled_continuous = scaler.transform(new_data[continuous_features])
    scaled_df = pd.DataFrame(scaled_continuous, columns=continuous_features)

    # Concaténation des caractéristiques encodées et normalisées
    processed_data = pd.concat([scaled_df, encoded_df], axis=1)

    return processed_data

def main():
    # Spécifiez le run_id du modèle et des préprocesseurs
    run_id = "2e509b4e47e84345b7824174f4c8947f"
    
    # Chargement du modèle
    model = load_model(run_id)
    
    # Chargement des préprocesseurs
    encoder, scaler = load_preprocessors(run_id)
    
    # Exemple de nouvelles données
    new_data = pd.DataFrame({
        'FlightNumber': ['AB123'],
        'DepartureAirport': ['JFK'],
        'ArrivalAirport': ['LAX'],
        'DepartureCondition': ['Clear'],
        'ArrivalCondition': ['Sunny'],
        'DepartureTempC': [15.0],
        'DepartureHumidity': [70],
        'DeparturePrecipMM': [0.0],
        'DepartureWindKPH': [10],
        'DepartureVisKM': [10.0],
        'DepartureGustKPH': [20.0],
        'ArrivalTempC': [25.0],
        'ArrivalHumidity': [50],
        'ArrivalPrecipMM': [0.0],
        'ArrivalWindKPH': [15.0],
        'ArrivalVisKM': [10.0],
        'ArrivalGustKPH': [30.0],
        'DepartureHour': [14],
        'ArrivalHour': [16],
        'DepartureDayOfWeek': [1],  
        'ArrivalDayOfWeek': [1],    
        'DepartureMonth': [9],      
        'ArrivalMonth': [9]         
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
