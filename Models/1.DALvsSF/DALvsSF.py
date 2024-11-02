import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error

data = 'Models/1.DALvsSF/DakPrescott.csv' # Will need to change depending on where you have CSV
features = ['Cmp', 'Pass_TD', 'Y/A']
target = ['Pass_Yds']

#Opponent Average Defense stats this season https://www.pro-football-reference.com/teams/sfo/2024.html
opponent_games_played = 7

opp_pass_yards_allowed = (1436)/opponent_games_played
opp_pass_att_allowed = (260-38)/opponent_games_played
opp_pass_comp_allowed = (161-25)/opponent_games_played
opp_yardsAtt_allowed = (opp_pass_yards_allowed/opp_pass_att_allowed)
opp_passTD_allowed = (10-2)/opponent_games_played

# QB Average stats this season https://www.pro-football-reference.com/players/P/PresDa01.html
games_played = 6

passing_yards = (1602)/games_played
pass_attempts = (224)/games_played
pass_completions = (142)/games_played
pass_yardsPerAttempt = (passing_yards/pass_attempts)
pass_TDs = (8)/games_played

# QB Stats Normalized
attempts = (opp_pass_att_allowed/pass_attempts) * pass_attempts
completions = (opp_pass_comp_allowed/pass_completions) * pass_completions
yardsPerAtt = (opp_yardsAtt_allowed/pass_yardsPerAttempt) * pass_yardsPerAttempt
td = (opp_passTD_allowed/pass_TDs) * pass_TDs

# Predicted inputs
inputs = [completions, td, yardsPerAtt]
test = pd.DataFrame([inputs], columns=features)
print(test)


def prediction(data, features, target, norm1, norm2):
    df = read_data(data)

    X_train, X_test, y_train, y_test = split_data(df[features], df[target])

    model = train_model(X_train, y_train)
    test_model(model, X_test, y_test)

    # Convert prediction result to float
    final_predict = model.predict(test)
    final_predict = final_predict[0, 0]
    final_predict = float(final_predict)
    norm_predict = (norm1/norm2) * final_predict # Normalize

    # Display results
    print(f'Average Pass Yards This Season: {passing_yards:.2f}')
    print(f'Average Opponent Pass Yards Allowed 2024: {opp_pass_yards_allowed:.2f}')
    print(f'Predicted Pass Yards: {final_predict:.2f}')
    print(f'Predicted Normalized: {norm_predict:.2f}')


# Read data set
def read_data(data):
    df = pd.read_csv(data)
    df = df.dropna()
    return df

# Split data
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    reg = Ridge()
    reg.fit(X_train, y_train)
    return reg

# Test the model
def test_model(model, X_test, y_test):
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    maePer = mean_absolute_percentage_error(y_test, prediction) * 100
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    print(f'Mean Absolute Error: {mae:.5f}\nMAE%: {maePer:.2f} %\nMean Squared Error: {mse:.5f}\nR2 Score: {r2:.5f}')

# Execute
prediction(data, features, target, passing_yards, opp_pass_yards_allowed)