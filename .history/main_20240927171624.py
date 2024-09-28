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
# Modèle pour la création de nouvelles questions
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



    
    
    
    question: str = "Quelle est la couleur du ciel ?"
    subject: str = "Systèmes distribués"
    correct: List[str] = ["Réponse A"]
    use: str = "Test de validation"
    responseA: str = "Bleu"
    responseB: str = "Rouge"
    responseC: str = "Vert"
    responseD: str = None  # La réponse D est facultative


# Endpoint pour créer une nouvelle question (admin)
@app.post("/create_question", response_description="Question ajoutée avec succès")
def predict(flight : Flightcreate):
    FlightNumber = flight.FlightNumber
    DepartureAirport = flight.DepartureAirport


    """
    Permet à un administrateur d'ajouter une nouvelle question dans le fichier CSV.

    - **admin_username** : Nom d'utilisateur admin (ex : "****").
    - **admin_password** : Mot de passe admin (ex : "****").
    - **question** : La question à poser.
    - **subject** : Le sujet de la question (ex : "Systèmes distribués").
    - **correct** : Liste des réponses correctes (ex : ["Réponse A"]).
    - **use** : Le type de test dans lequel cette question sera utilisée.
    - **responseA, responseB, responseC, responseD** : Les options de réponse (réponse D facultative).
    """
    if question.admin_username != "admin" or question.admin_password != "4dm1N":
        raise HTTPException(
            status_code=403, detail="Identifiants administrateur incorrects.")

    # Ajouter la question au fichier CSV
    with open('questions.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question', 'subject', 'correct', 'use',
                      'responseA', 'responseB', 'responseC', 'responseD']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'question': question.question,
            'subject': question.subject,
            'correct': ','.join(question.correct),
            'use': question.use,
            'responseA': question.responseA,
            'responseB': question.responseB,
            'responseC': question.responseC,
            'responseD': question.responseD
        })

    return {"message": "Question ajoutée avec succès."}