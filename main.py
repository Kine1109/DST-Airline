from fastapi import FastAPI, HTTPException,Request
from pydantic import BaseModel
from pymongo import MongoClient
from models.predict import main
import pandas as pd
import os 



# Configuration MongoDB
MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = 'flight_data'
COLLECTION_NAME = 'flights_with_weather'

CLIENT_MONGO = MongoClient(MONGO_URI)
DB = CLIENT_MONGO[DB_NAME]
COLLECTION = DB[COLLECTION_NAME]

# Initialisation de l'API
api = FastAPI()

@api.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Request URL: {request.url.path}")
    response = await call_next(request)
    return response

# Modèle de données pour la création de prédictions
class FlightCreate(BaseModel):
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

# Endpoint pour créer une nouvelle prédiction
@api.post("/predict")
def predict(flight: FlightCreate):
    print('ok')
    print(flight)
    try:
        # Convertir les données en DataFrame
        new_data = pd.DataFrame([flight.dict()])
        result = main(new_data)
        return {"prediction_delay": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")

