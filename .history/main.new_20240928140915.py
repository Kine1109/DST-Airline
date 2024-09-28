from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from bson import json_util  # Pour gérer la sérialisation des objets BSON
from models.predict import main
import pandas as pd



# Configuration MongoDB
MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority'
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

CLIENT_MONGO = MongoClient(MONGO_URI)
DB = CLIENT_MONGO[DB_NAME]
COLLECTION = DB[COLLECTION_NAME]

# Initialisation de l'API
api = FastAPI()

# Modèle de données pour la création de prédictions
class FlightCreate(BaseModel):
    FlightNumber: str
    DepartureAirport: str
    ArrivalAirport: str
    DepartureCondition: str
    ArrivalCondition: str
    DepartureTempC: float
    DepartureHumidity: int
    DeparturePrecipMM: float
    DepartureWindKPH: float
    DepartureVisKM: float
    DepartureGustKPH: float
    ArrivalTempC: float
    ArrivalHumidity: int
    ArrivalPrecipMM: float
    ArrivalWindKPH: float
    ArrivalVisKM: float
    ArrivalGustKPH: float
    DepartureHour: int
    ArrivalHour: int
    DepartureDayOfWeek: int
    ArrivalDayOfWeek: int
    DepartureMonth: int
    ArrivalMonth: int


# Endpoint racine pour vérifier le fonctionnement de l'API
@api.get("/")
def root():
    return {"message": "Bienvenue sur l'API de prédiction de retards de vols !"}

# Endpoint pour télécharger les données de vols depuis MongoDB
@api.get('/FlightData')
def get_data():
    flights = COLLECTION.find({})
    flights = list(flights)  # Convertir le curseur en liste de dictionnaires
    return json_util.dumps(flights)

# Endpoint pour obtenir les données d'un vol spécifique par FlightNumber
@api.get('/FlightData_FlightNumber/{FlightNumber}')
def get_data_flight_number(FlightNumber: str):
    flights = COLLECTION.find({'FlightNumber': FlightNumber})
    flights = list(flights)  # Convertir le curseur en liste de dictionnaires
    return json_util.dumps(flights)

# Endpoint pour créer une nouvelle prédiction
@api.post("/predict")
def predict(flight: FlightCreate):
    # Convertir les données en DataFrame
    new_data = pd.DataFrame([flight.dict()])

    # Utiliser la fonction main() importée depuis predict.py pour faire une prédiction
    try:
        result = main(new_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")
