import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

data = 'Models/2.DALvsATL/DakPrescott.csv' # Will need to change depending on where you have CSV
features = ['Cmp', 'Pass_TD', 'Y/A']
target = ['Pass_Yds']

#Opp Defense Data Current (ATL) https://www.pro-football-reference.com/teams/atl/2024.htm
games_played = 8

opp_passYards_allowed = (1724)/games_played
opp_cmp_allowed = (195)/games_played
opp_passAtt_allowed = (269)/games_played

opp_passTD_allowed = (13)/games_played

opp_ya = (opp_passYards_allowed/opp_passAtt_allowed)

#Qb vs Opp Defense Data https://www.pro-football-reference.com/players/P/PresDa01/gamelog/?opp_id=atl
qb_gamesPlayed_opp = 4

qb_passYards_opp = (1130)/qb_gamesPlayed_opp

qb_cmp_opp = (100)/qb_gamesPlayed_opp
qb_att_opp = (140)/qb_gamesPlayed_opp

qb_passTD_opp = (3)/qb_gamesPlayed_opp
qb_ya_opp = 8.07

# QB Stats Normalized
completions = (qb_cmp_opp/opp_cmp_allowed) * qb_cmp_opp
attempts = (qb_att_opp/opp_passAtt_allowed) * qb_att_opp

passTD = (qb_passTD_opp/opp_passTD_allowed) * qb_passTD_opp

yardsPerAttempt = (qb_ya_opp/opp_ya) * qb_ya_opp

# Predicted inputs
inputs = [completions, passTD, yardsPerAttempt]
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
    print(f'Average Pass Yards vs Opponent: {qb_passYards_opp:.2f}')
    print(f'Average Opponent Pass Yards Allowed 2024: {opp_passYards_allowed:.2f}')
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
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    return linear_reg

# Test the model
def test_model(model, X_test, y_test):
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    maePer = mean_absolute_percentage_error(y_test, prediction) *100
    r2 = r2_score(y_test, prediction)
    print(f'Mean Absolute Error: {mae:.2f}\nMAE %: {maePer:.2f}\nR2 Score: {r2:.2f}')

# Execute
prediction(data, features, target, qb_passYards_opp, opp_passYards_allowed)