import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


using = fetch_california_housing()

data = pd.DataFrame(housing.data,
columns = housing.feature_names)
data["Price"] = housing.target



x = data[['AveRooms']].values
y=data['Price'].values



x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    test_size=0.2,
    random_state=42
)



scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)



w=0
b=0

learning_rate = 0.01
epochs = 1000

n=len(x_train_scaled)

for i in range(epochs):
    y_pred = w * x_train_scaled.flatten() + b
    dw = (1/n) * np.sum((y_pred - y_train) * x_train_scaled.flatten())
    db = (1/n) * np.sum(y_pred - y_train)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    if i % 100 ==0:
        cost = (1/(2*n)) * np.sum((y_pred - y_train) **2)
        print(f"Epoch {i}, Cost = {cost:.4f}")
y_pred_gd = w * x_test_scaled.flatten() + b



*************************Output********************************
Epoch 0, Cost = 2.8149
Epoch 100, Cost = 0.9414
Epoch 200, Cost = 0.6904
Epoch 300, Cost = 0.6568
Epoch 400, Cost = 0.6523
Epoch 500, Cost = 0.6517
Epoch 600, Cost = 0.6516
Epoch 700, Cost = 0.6516
Epoch 800, Cost = 0.6516
Epoch 900, Cost = 0.6516
***************************************************************
