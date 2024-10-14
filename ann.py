import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'player_id': np.random.randint(0, 100, 1000),
    'team_id': np.random.randint(0, 30, 1000),
    'opponent_team_id': np.random.randint(0, 30, 1000),
    'home_away': np.random.randint(0, 2, 1000),
    'player_minutes': np.random.randint(10, 40, 1000),
    'player_stat': np.random.randint(0, 50, 1000)
})

features = ['player_id', 'team_id', 'opponent_team_id', 'home_away', 'player_minutes']
target = 'player_points'  

X = data[features].values
y = data[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

#model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
optimizer = 'adam'(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
loss, mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {mae}')

predictions = model.predict(X_test)
