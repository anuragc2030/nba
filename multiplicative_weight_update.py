import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Dense


file_path = r'C:\Users\anura\Downloads\2023-2024 NBA Player Stats - Playoffs.csv'
df_nba_stats_playoffs = pd.read_csv(file_path, delimiter=';')
X = df_nba_stats_playoffs[['G', 'MP', 'FGA', '3PA', '2PA', 'FTA', 'TRB', 'STL','PTS']]

y = df_nba_stats_playoffs['PTS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize model weights
weights = np.array([1.0, 1.0, 1.0, 1.0])
iterations = 10

linear = linear_model.LinearRegression()

model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_nn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model_nn.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

model_dt = DecisionTreeRegressor(random_state=42)
model_dt.fit(X_train, y_train)
linear.fit(X_train, y_train)
pred_linear = linear.predict(X_test)
loss_linear = mean_squared_error(y_test, pred_linear)

# Train Model 2 
pred_nn = model_nn.predict(X_test).flatten()
loss_nn = mean_absolute_error(y_test, pred_nn)

# Train Model 3 
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
loss_rf = mean_squared_error(y_test, pred_rf)

 # Train Model 4 
model_dt.fit(X_train, y_train)
pred_dt = model_dt.predict(X_test)
loss_dt = mean_squared_error(y_test, pred_dt)

# Compute the absolute errors for each prediction
errors_linear = np.abs(y_test - pred_linear)
errors_nn = np.abs(y_test - pred_nn)
errors_rf = np.abs(y_test - pred_rf)
errors_dt = np.abs(y_test - pred_dt)

# Update weights
weights[0] *= np.prod(1 / (2 ** errors_linear))
weights[1] *= np.prod(1 / (2 ** errors_nn))
weights[2] *= np.prod(1 / (2 ** errors_rf))
weights[3] *= np.prod(1 / (2 ** errors_dt))

weights /= weights.sum()

final_predictions = (weights[0] * pred_linear + 
                     weights[1] * pred_nn + 
                     weights[2] * pred_rf + 
                     weights[3] * pred_dt)/4

# Evaluate the model
final_mae = mean_squared_error(y_test, final_predictions)
print(f'Iteration {i+1}, Final Test MSE: {final_mae}')

print(f'Final Weights: {weights}')