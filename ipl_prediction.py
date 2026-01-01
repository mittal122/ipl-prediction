"""
IPL First Innings Score Prediction
This script trains machine learning models to predict IPL first innings scores
"""

# Importing essential libraries
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

print("=" * 50)
print("IPL First Innings Score Prediction")
print("=" * 50)

# Loading the dataset
print("\n1. Loading the dataset...")
df = pd.read_csv('dataset/ipl.csv')
print(f"Dataset loaded successfully! Shape: {df.shape}")

# Exploring the dataset
print("\n2. Exploring the dataset...")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
print(f"\nFirst few rows:\n{df.head()}")

# Data Cleaning
print("\n" + "=" * 50)
print("3. DATA CLEANING")
print("=" * 50)

# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
print(f'\nBefore removing unwanted columns: {df.shape}')
df.drop(labels=columns_to_remove, axis=1, inplace=True)
print(f'After removing unwanted columns: {df.shape}')

# Keeping only consistent teams
print(f"\nUnique batting teams: {df['bat_team'].unique()}")
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']

print(f'Before removing inconsistent teams: {df.shape}')
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]
print(f'After removing inconsistent teams: {df.shape}')

# Removing the first 5 overs data in every match
print(f'\nBefore removing first 5 overs data: {df.shape}')
df = df[df['overs']>=5.0]
print(f'After removing first 5 overs data: {df.shape}')

# Converting the column 'date' from string into datetime object
print(f"\nBefore converting 'date' column: {type(df.iloc[0,0])}")
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print(f"After converting 'date' column: {type(df.iloc[0,0])}")

# Data Preprocessing
print("\n" + "=" * 50)
print("4. DATA PREPROCESSING")
print("=" * 50)

# Converting categorical features using OneHotEncoding method
print("\nApplying One-Hot Encoding to categorical features...")
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
print(f"Encoded columns: {list(encoded_df.columns)}")

# Rearranging the columns
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
              'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
              'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# Splitting the data into train and test set
print("\nSplitting dataset into train (2008-2016) and test (2017+) sets...")
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

print(f"Training set: {X_train.shape} and Test set: {X_test.shape}")

# Model Building
print("\n" + "=" * 50)
print("5. MODEL BUILDING")
print("=" * 50)

# Linear Regression Model
print("\n--- Linear Regression ---")
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_lr = linear_regressor.predict(X_test)

print("---- Linear Regression - Model Evaluation ----")
print(f"Mean Absolute Error (MAE): {mae(y_test, y_pred_lr):.2f}")
print(f"Mean Squared Error (MSE): {mse(y_test, y_pred_lr):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse(y_test, y_pred_lr)):.2f}")

# Decision Tree Regression Model
print("\n--- Decision Tree Regression ---")
decision_regressor = DecisionTreeRegressor()
decision_regressor.fit(X_train, y_train)
y_pred_dt = decision_regressor.predict(X_test)

print("---- Decision Tree Regression - Model Evaluation ----")
print(f"Mean Absolute Error (MAE): {mae(y_test, y_pred_dt):.2f}")
print(f"Mean Squared Error (MSE): {mse(y_test, y_pred_dt):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse(y_test, y_pred_dt)):.2f}")

# Random Forest Regression Model
print("\n--- Random Forest Regression ---")
random_regressor = RandomForestRegressor()
random_regressor.fit(X_train, y_train)
y_pred_rf = random_regressor.predict(X_test)

print("---- Random Forest Regression - Model Evaluation ----")
print(f"Mean Absolute Error (MAE): {mae(y_test, y_pred_rf):.2f}")
print(f"Mean Squared Error (MSE): {mse(y_test, y_pred_rf):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse(y_test, y_pred_rf)):.2f}")

# AdaBoost Model using Linear Regression as the base learner
print("\n--- AdaBoost Regression ---")
adb_regressor = AdaBoostRegressor(estimator=linear_regressor, n_estimators=100)
adb_regressor.fit(X_train, y_train)
y_pred_adb = adb_regressor.predict(X_test)

print("---- AdaBoost Regression - Model Evaluation ----")
print(f"Mean Absolute Error (MAE): {mae(y_test, y_pred_adb):.2f}")
print(f"Mean Squared Error (MSE): {mse(y_test, y_pred_adb):.2f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse(y_test, y_pred_adb)):.2f}")

# Prediction Function
def predict_score(batting_team='Chennai Super Kings', bowling_team='Mumbai Indians', 
                  overs=5.1, runs=50, wickets=0, runs_in_prev_5=50, wickets_in_prev_5=0):
    """
    Predict the final score of first innings based on current match situation
    """
    temp_array = list()

    # Batting Team
    if batting_team == 'Chennai Super Kings':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif batting_team == 'Delhi Daredevils':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif batting_team == 'Kings XI Punjab':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif batting_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif batting_team == 'Mumbai Indians':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif batting_team == 'Rajasthan Royals':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif batting_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif batting_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    # Bowling Team
    if bowling_team == 'Chennai Super Kings':
        temp_array = temp_array + [1,0,0,0,0,0,0,0]
    elif bowling_team == 'Delhi Daredevils':
        temp_array = temp_array + [0,1,0,0,0,0,0,0]
    elif bowling_team == 'Kings XI Punjab':
        temp_array = temp_array + [0,0,1,0,0,0,0,0]
    elif bowling_team == 'Kolkata Knight Riders':
        temp_array = temp_array + [0,0,0,1,0,0,0,0]
    elif bowling_team == 'Mumbai Indians':
        temp_array = temp_array + [0,0,0,0,1,0,0,0]
    elif bowling_team == 'Rajasthan Royals':
        temp_array = temp_array + [0,0,0,0,0,1,0,0]
    elif bowling_team == 'Royal Challengers Bangalore':
        temp_array = temp_array + [0,0,0,0,0,0,1,0]
    elif bowling_team == 'Sunrisers Hyderabad':
        temp_array = temp_array + [0,0,0,0,0,0,0,1]

    # Overs, Runs, Wickets, Runs_in_prev_5, Wickets_in_prev_5
    temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]

    # Converting into numpy array
    temp_array = np.array([temp_array])

    # Prediction
    return int(linear_regressor.predict(temp_array)[0])

# Sample Predictions
print("\n" + "=" * 50)
print("6. SAMPLE PREDICTIONS")
print("=" * 50)

predictions = [
    {
        'name': 'Prediction 1',
        'date': '16th April 2018',
        'season': 11,
        'match': 13,
        'batting': 'Kolkata Knight Riders',
        'bowling': 'Delhi Daredevils',
        'actual': 200,
        'overs': 9.2,
        'runs': 79,
        'wickets': 2,
        'runs_in_prev_5': 60,
        'wickets_in_prev_5': 1
    },
    {
        'name': 'Prediction 2',
        'date': '7th May 2018',
        'season': 11,
        'match': 39,
        'batting': 'Sunrisers Hyderabad',
        'bowling': 'Royal Challengers Bangalore',
        'actual': 146,
        'overs': 10.5,
        'runs': 67,
        'wickets': 3,
        'runs_in_prev_5': 29,
        'wickets_in_prev_5': 1
    },
    {
        'name': 'Prediction 3',
        'date': '17th May 2018',
        'season': 11,
        'match': 50,
        'batting': 'Mumbai Indians',
        'bowling': 'Kings XI Punjab',
        'actual': 186,
        'overs': 14.1,
        'runs': 136,
        'wickets': 4,
        'runs_in_prev_5': 50,
        'wickets_in_prev_5': 0
    },
    {
        'name': 'Prediction 4',
        'date': '30th March 2019',
        'season': 12,
        'match': 9,
        'batting': 'Mumbai Indians',
        'bowling': 'Kings XI Punjab',
        'actual': 176,
        'overs': 12.3,
        'runs': 113,
        'wickets': 2,
        'runs_in_prev_5': 55,
        'wickets_in_prev_5': 0
    },
    {
        'name': 'Prediction 5',
        'date': '11th April 2019',
        'season': 12,
        'match': 25,
        'batting': 'Rajasthan Royals',
        'bowling': 'Chennai Super Kings',
        'actual': 151,
        'overs': 13.3,
        'runs': 92,
        'wickets': 5,
        'runs_in_prev_5': 27,
        'wickets_in_prev_5': 2
    },
    {
        'name': 'Prediction 6',
        'date': '14th April 2019',
        'season': 12,
        'match': 30,
        'batting': 'Delhi Daredevils',
        'bowling': 'Sunrisers Hyderabad',
        'actual': 155,
        'overs': 11.5,
        'runs': 98,
        'wickets': 3,
        'runs_in_prev_5': 41,
        'wickets_in_prev_5': 1
    },
    {
        'name': 'Prediction 7',
        'date': '10th May 2019',
        'season': 12,
        'match': 59,
        'batting': 'Delhi Daredevils',
        'bowling': 'Chennai Super Kings',
        'actual': 147,
        'overs': 10.2,
        'runs': 68,
        'wickets': 3,
        'runs_in_prev_5': 29,
        'wickets_in_prev_5': 1
    }
]

for pred in predictions:
    final_score = predict_score(
        batting_team=pred['batting'],
        bowling_team=pred['bowling'],
        overs=pred['overs'],
        runs=pred['runs'],
        wickets=pred['wickets'],
        runs_in_prev_5=pred['runs_in_prev_5'],
        wickets_in_prev_5=pred['wickets_in_prev_5']
    )
    print(f"\n{pred['name']} ({pred['date']}, IPL Season {pred['season']})")
    print(f"Match: {pred['batting']} vs {pred['bowling']}")
    print(f"Actual Score: {pred['actual']}")
    print(f"Predicted Score Range: {final_score-10} to {final_score+5}")

print("\n" + "=" * 50)
print("Note: In IPL, it is very difficult to predict the actual score")
print("because the game can completely turn upside down in a moment!")
print("=" * 50)
