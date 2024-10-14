#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Define the file path, ensure to use double backslashes or raw string to avoid issues with escape characters
file_path = r'C:\Users\anura\Downloads\2023-2024 NBA Player Stats - Playoffs.csv'

# Load the CSV file into a DataFrame
df_nba_stats_playoffs = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the DataFrame to verify loading
print(df_nba_stats_playoffs.head())

# Define the features and target variable
X = df_nba_stats_playoffs[['MP', 'FGA', '3PA', '2PA', 'FTA', 'TRB', 'STL']]
y = df_nba_stats_playoffs['PTS']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize and train the Linear Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get player names for the test set
player_names = df_nba_stats_playoffs.loc[y_test.index, 'Player'][:70].values

# Display the first ten predictions and the first ten actual results along with player names and games played
results = pd.DataFrame({
    'Player': player_names,
    'Games Played': df_nba_stats_playoffs.loc[y_test.index, 'G'][:70].values,
    'Predicted Points': y_pred[:70],
    'Actual Points': y_test[:70].values
})
print(results)

#mse = 10000000 
#model = None
#for depth in depths :|
#   for estimator is n_estimators:
#       randomforest = rforest(depth,estimator) 
#       trainforest 
#       testforest on val set
#       if MSE on val set < mse :
#           mse = MSE_val_Set 
#           model = rforest
