# -*- coding: utf-8 -*-
"""PE_Assignment1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KVn20KPUvN6FFHchPa4k92KxhD5tS9AV
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

car_data =pd.read_csv('car data.csv')

car_data.head()

car_data.info()

car_data['Year'].astype(float).dtypes

car_data['Kms_Driven'].astype(float).dtypes

car_data.describe()

car_data.shape

car_data.columns

car_data['Fuel_Type'].unique()

car_data['Fuel_Type'].replace({'Petrol':0,'Diesel':1,'CNG':2},inplace=True)

car_data['Fuel_Type'].unique()

car_data['Present_Price'].astype(int).dtypes

plt.scatter(car_data['Selling_Price'],car_data['Present_Price'])
plt.xlabel('Selling_price')
plt.ylabel('Present_Price')
plt.title('Scatter Plot: Prediction')
plt.show()

"""# Linear Regression"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score,accuracy_score
from sklearn.linear_model import LinearRegression
X =car_data[['Year','Selling_Price','Kms_Driven','Fuel_Type']]
y=car_data['Present_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Shape of y_test:", y_test.shape)

print("Shape of pred:", pred.shape)

mse = mean_squared_error(y_test,pred)
mae = mean_absolute_error(y_test,pred)
r2 = r2_score(y_test, pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

model.score(X_train,y_train)

sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()



"""# Decision Tree"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X =car_data[['Year','Selling_Price','Kms_Driven','Fuel_Type']]
y=car_data['Present_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

predicted_price = model.predict(X_test)

print("Predicted Price:", predicted_price)

print("Shape of y_test:", y_test.shape)

print("Shape of pred:", pred.shape)

mse = mean_squared_error(y_test,pred)
mae = mean_absolute_error(y_test,pred)
r2 = r2_score(y_test, pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

model.score(X_train,y_train)

sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()

"""# Random Forests"""

from sklearn.ensemble import RandomForestRegressor

X =car_data[['Year','Selling_Price','Kms_Driven','Fuel_Type']]
y=car_data['Present_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)
y_pred = model.predict(X_test)

predicted_price = model.predict(X_test)

print("Predicted Price:", predicted_price)

mse = mean_squared_error(y_test,pred)
mae = mean_absolute_error(y_test,pred)
r2 = r2_score(y_test, pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

model.score(X_train,y_train)

sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()

"""# KNN"""

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

X =car_data[['Year','Selling_Price','Kms_Driven','Fuel_Type']]
y=car_data['Present_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = 5  # Number of neighbors
knn_model = KNeighborsRegressor(n_neighbors=k)
knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)

mse = mean_squared_error(y_test,pred)
mae = mean_absolute_error(y_test,pred)
r2 = r2_score(y_test, pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

sns.regplot(x=pred, y=y_test)
plt.xlabel("Predicted Price")
plt.ylabel('Actual Price')
plt.title("ACtual vs predicted price")
plt.show()