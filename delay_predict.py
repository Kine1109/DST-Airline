import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from pymongo import MongoClient

data = pd.read_csv('flight_delays.csv',header = 0)
flights_df = data[['FlightNumber','Origin','Destination','DelayMinutes','DelayReason']]
# Préparation des données

# Extraire l'heure, le jour de la semaine, et le mois des dates locales
flights_df['DepartureHour'] = pd.to_datetime(data['ActualDeparture']).dt.hour
flights_df['ArrivalHour'] = pd.to_datetime(data['ActualArrival']).dt.hour
flights_df['DepartureDayOfWeek'] = pd.to_datetime(data['ActualDeparture']).dt.dayofweek
flights_df['ArrivalDayOfWeek'] = pd.to_datetime(data['ActualArrival']).dt.dayofweek
flights_df['DepartureMonth'] = pd.to_datetime(data['ActualDeparture']).dt.month
flights_df['ArrivalMonth'] = pd.to_datetime(data['ActualArrival']).dt.month
#print(flights_df[flights_df['DelayReason'] == 'Weather'])
print(flights_df['DelayReason'])
# Encodage des variables catégorielles (FlightNumber, Airports, etc.)
#categorical_features = ['FlightNumber', 'DepartureAirport', 'ArrivalAirport', 'DepartureCondition', 'ArrivalCondition']
categorical_features = ['FlightNumber', 'Origin', 'Destination', 'DelayReason']

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(flights_df[categorical_features])

# Normalisation des caractéristiques continues (temp_c, humidity, etc.)
"""
continuous_features = ['DepartureTempC', 'DepartureHumidity', 'DeparturePrecipMM', 
                       'DepartureWindKPH', 'DepartureVisKM', 'ArrivalGustKPH',
                       'ArrivalTempC', 'ArrivalHumidity', 'ArrivalPrecipMM', 
                       'ArrivalWindKPH', 'ArrivalVisKM', 'ArrivalGustKPH',
                       'DepartureHour', 'ArrivalHour', 'DepartureDayOfWeek', 'ArrivalDayOfWeek', 
                       'DepartureMonth', 'ArrivalMonth']
"""
continuous_features = ['DepartureHour', 'ArrivalHour', 'DepartureDayOfWeek', 'ArrivalDayOfWeek', 
                       'DepartureMonth', 'ArrivalMonth']
scaler = StandardScaler()
scaled_continuous = scaler.fit_transform(flights_df[continuous_features])

# Fusionner les données encodées et normalisées
X = np.hstack([encoded_categorical, scaled_continuous])
print(X.shape)
# La variable cible à prédire : Retard à l'arrivée
y = flights_df['DelayMinutes'].values

# 2. Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Construction du modèle de deep learning
model = Sequential()

# Ajout d'une couche d'entrée (input layer)
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))

# Ajout d'une couche cachée
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Ajout de la couche de sortie
model.add(Dense(1, activation='linear'))  # Activation linéaire pour prédiction de valeurs continues

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 4. Entraînement du modèle
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# 5. Évaluation du modèle
loss, mae = model.evaluate(X_test, y_test)
print(f'Mean Absolute Error on Test Set: {mae}')
