import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



data = pd.read_csv('car data.csv')

print(data.head())
#This typically insprects the data

print(data.isnull().sum())
#Check for the missing values


data = data.dropna()
# This handels the missing values if found any

X = data[['year', 'milage', 'engine_size', 'horse_power']]
y = data['price']
# feature selection

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Split the data into training and testing sets

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Standardize the feature

model = RandomForestRegressor(n_estimators=100, random_state=42)
#Initialize the model

model.fit(X_train, y_train)
# Train the model

y_pred = model.predict(X_test)
# Make prediction on the test case

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
# Evaluate the model for the cases

print("Mean Absolute Error:", mae)
print("Mean Squared Error", mse)
print("Root Mean Squared Error:", rmse)

import joblib
joblib.dump(model, 'car_prediction _model.pkl')
# Save the model to the file
# Load the model for future use
# loaded_model = joblib.load('car_prediction_model.pkl')

# Predicting a new car price
new_car = [[2020, 15000, 2.0, 150]]
new_car = scaler.transform(new_car)
predicted_price = model.predict(new_car_scaled)
print("Predicted price for the new car: ", predicted_price)