#import libraries
#EDA
#split the dataset into test and train sets
#build the model
#train the model
#evaluate model
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error


# Define the file path, ensure to use double backslashes or raw string to avoid issues with escape characters
file_path = r'C:\Users\anura\Downloads\2023-2024 NBA Player Stats - Playoffs.csv'

# Load the CSV file into a DataFrame
df_nba_stats_playoffs = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the DataFrame to verify loading
print(df_nba_stats_playoffs.head())
X = df_nba_stats_playoffs[['G', 'MP', 'FGA', '3PA', '2PA', 'FTA', 'TRB', 'STL']]
y = df_nba_stats_playoffs[['G', 'MP', 'FGA', '3PA', '2PA', 'FTA', 'TRB', 'STL']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MAE: {mae}')



