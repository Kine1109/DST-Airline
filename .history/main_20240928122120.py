from fastapi import FastAPI,HTTPException, Request
from pydantic import BaseModel
import base64
from typing import List
import pandas as pd
from pymongo import MongoClient
from bson import json_util  # Pour gérer la sérialisation des objets BSON
from models.predict import main


MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority' 
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

CLIENT_MONGO = MongoClient(MONGO_URI)
DB = CLIENT_MONGO[DB_NAME]
COLLECTION = DB[COLLECTION_NAME]


api = FastAPI()

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
        'DepartureHour': [0], #int
        'ArrivalHour': [17],
        'DepartureDayOfWeek': [5],  
        'ArrivalDayOfWeek': [5],    
        'DepartureMonth': [8],      
        'ArrivalMonth': [8],
         'De': [8]         
    })


@api.get("/")
def root():
    #
    #
    return json_util.dumps(main(new_data))

# fontion pour télécharger les données de vols depuis MongoDB
@api.get('/FlightData')
def get_data():
    flights = COLLECTION.find({})
    # Convertir le curseur en liste de dictionnaires
    flights = list(flights)
    
    # Utiliser json_util pour sérialiser les objets BSON en JSON
    return json_util.dumps(flights)

@api.get('/FlightData_FlightNumber/{FlightNumber}')
def get_Data_FlightNumber(FlightNumber):
    flights = COLLECTION.find({'FlightNumber':FlightNumber})
    # Convertir le curseur en liste de dictionnaires
    flights = list(flights)
    
    # Utiliser json_util pour sérialiser les objets BSON en JSON
    return json_util.dumps(flights)


#------------------------------------ RS
# Modèle pour la création de nouvelles predictions
class Flightcreate(BaseModel):
    FlightNumber: str
    DepartureAirport: str
    ArrivalAirport : str
    DepartureCondition : str
    ArrivalCondition : str
    DepartureTempC : float
    DepartureHumidity : float
    DeparturePrecipMM : float
    DepartureWindKPH : float
    DepartureVisKM : float
    DepartureGustKPH : float
    ArrivalTempC : float
    ArrivalHumidity : int
    ArrivalPrecipMM : float
    ArrivalWindKPH : float
    ArrivalVisKM : float
    ArrivalGustKPH : float
    DepartureHour : int
    ArrivalHour : int
    DepartureDayOfWeek : int
    ArrivalDayOfWeek : int
    DepartureMonth : int
    ArrivalMonth : int
    De : int
  


# Endpoint pour créer une nouvelle question (admin)
@app.post("/create_question")
def predict(flight: Flightcreate):
    FlightNumber = flight.FlightNumber
    DepartureAirport = flight.DepartureAirport
    ArrivalAirport = flight.ArrivalAirport
    DepartureCondition = flight.DepartureCondition
    ArrivalCondition = flight.ArrivalCondition
    DepartureTempC = flight.DepartureTempC
    DepartureHumidity = flight.DepartureHumidity
    DeparturePrecipMM = flight.DeparturePrecipMM
    DepartureWindKPH = flight.DepartureWindKPH
    DepartureVisKM = flight.DepartureVisKM
    DepartureGustKPH = flight.DepartureGustKPH
    ArrivalTempC = flight.ArrivalTempC
    ArrivalHumidity = flight.ArrivalHumidity
    ArrivalPrecipMM = flight.ArrivalPrecipMM
    ArrivalWindKPH = flight.ArrivalWindKPH
    ArrivalVisKM = flight.ArrivalVisKM
    ArrivalGustKPH = flight.ArrivalGustKPH
    DepartureHour = flight.DepartureHour
    ArrivalHour = flight.ArrivalHour
    DepartureDayOfWeek = flight.DepartureDayOfWeek
    ArrivalDayOfWeek = flight.ArrivalDayOfWeek
    DepartureMonth = flight.DepartureMonth
    ArrivalMonth = flight.ArrivalMonth
    De = flight.De
