import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('car data.csv')

print(data.head())

print(data.isnull().sum())

data = data.dropna()

data['Fuel_Type'] = data['Fuel_Type'].map({'Petrol': 0, 'Diesel': 1, 'CNG': 2})
data['Transmission'] = data['Transmission'].map({'Manual': 0, 'Automatic': 1})

X = data[['Year', 'Driven_kms', 'Present_Price', 'Fuel_Type', 'Transmission']]
y = data['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

import joblib
joblib.dump(model, 'car_price_model.pkl')

new_car = [[2020, 15000, 5.59, 0, 0]]  
new_car_scaled = scaler.transform(new_car)
predicted_price = model.predict(new_car_scaled)
print("Predicted price for the new car:", predicted_price)
