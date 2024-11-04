import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error

# https://www.pro-football-reference.com/players/A/AlleJo02/gamelog/
data = 'Models/Josh Allen/JAllen_Regular_Passing.csv' # Will need to change depending on where you have CSV
features = ['Age', 'Cmp', 'Att', 'Cmp%', 'TD', 'Int', 'Rate', 'Sk', 'Y/A']
target = ['Yds']

# Defense AVG STATS THIS SEASON: https://www.pro-football-reference.com/teams/clt/2024.htm#all_defense
games_played =  9

pass_att_allowed = (291)/games_played
pass_cmp_allowed = (204)/games_played
passYds_allowed = (2104)/games_played
passTD_allowed = (14)/games_played
def_int = (6)/games_played
def_sks = (20)/games_played

# QB AVG STATS THIS SEASON: https://www.pro-football-reference.com/players/A/AlleJo02/gamelog/2024/

games_started = 9

qb_age = 28.185 # Nov 3rd 2024
qb_cmp = (168)/games_started
qb_att = (262)/games_started
qb_yds = (2001)/games_started
qb_TD = (17)/games_started
qb_int = (2)/games_started
qb_rate = 105.8
qb_sk = 11/games_started

# Normalizing stats
att = (pass_att_allowed/qb_att) * qb_att
cmp = (pass_cmp_allowed/qb_cmp) * qb_cmp
cmpPer = (cmp/att) * 100
td = (passTD_allowed/qb_TD) * qb_TD
ints = (def_int/qb_int) * qb_int
sacks = (def_sks/qb_sk) * qb_sk
ya = ((passYds_allowed/qb_yds) * qb_yds) / att


# Predicted inputs
input_stats = [qb_age, cmp, att, cmpPer, td, ints, qb_rate, sacks, ya]
test = pd.DataFrame([input_stats], columns=features)
print(test)

def prediction(data, features, target):
    df = read_data(data)

    X_train, X_test, y_train, y_test = split_data(df[features], df[target])

    model = train_model(X_train, y_train)
    test_model(model, X_test, y_test)

    # Convert prediction result to float
    final_predict = model.predict(test)
    final_predict = final_predict[0]
    final_predict = float(final_predict)

    # Display results
    print(f'Average Pass Yards This Season: {qb_yds:.2f}')
    print(f'Average Opponent Pass Yards Allowed 2024: {passYds_allowed:.2f}')
    print(f'Predicted Pass Yards: {final_predict:.2f}')


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
    reg = Lasso(alpha=1)
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
prediction(data, features, target)