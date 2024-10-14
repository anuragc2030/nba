#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Define the file path, ensure to use double backslashes or raw string to avoid issues with escape characters
file_path = r'C:\Users\anura\Downloads\2023-2024 NBA Player Stats - Playoffs.csv'

# Load the CSV file into a DataFrame
df_nba_stats_playoffs = pd.read_csv(file_path, delimiter=';')

# Display the first few rows of the DataFrame to verify loading
print(df_nba_stats_playoffs.head())

# Define the features and target variable
X = df_nba_stats_playoffs[['G', 'MP', 'FGA', '3PA', '2PA', 'FTA', 'TRB', 'STL']]
y = df_nba_stats_playoffs['PTS']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Display the first ten predictions and the first ten actual results
results = pd.DataFrame({'Predicted': y_pred[:10], 'Actual': y_test[:10].values})
print(results)
