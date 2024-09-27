from fastapi import FastAPI,HTTPException, Request
from pydantic import BaseModel
import base64
from typing import List
import pandas as pd
from pymongo import MongoClient
from bson import json_util  # Pour gérer la sérialisation des objets BSON



MONGO_URI = 'mongodb+srv://dst-airline-MRFF:gVlxqqz76838njKp@cluster0.vauxcgo.mongodb.net/test?retryWrites=true&w=majority' 
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

CLIENT_MONGO = MongoClient(MONGO_URI)
DB = CLIENT_MONGO[DB_NAME]
COLLECTION = DB[COLLECTION_NAME]


api = FastAPI()


@api.get("/")
def root():
    return {"message": "Hello, World!"}

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
