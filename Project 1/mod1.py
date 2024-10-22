import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

data = 'DakPrescott.csv'
features = ['Age', 'Cmp', 'Att', 'Cmp%', 'Pass_TD', 'Int', 'Passer_rating', 'Sk', 'Y/A']
target = ['Pass_Yds']

#Opp Defense Data
games_played = 7

opp_passYards_allowed = (1436)/games_played
opp_cmp_allowed = (136)/games_played
opp_passAtt_allowed = (222)/games_played
opp_cmpPer_allowed = ((opp_cmp_allowed/opp_passAtt_allowed) *100)
opp_passTD_allowed = (9)/games_played
opp_int = (8)/games_played
a = ((opp_cmp_allowed/opp_passAtt_allowed)-0.3) * 5
b = ((opp_passYards_allowed/opp_passAtt_allowed)-3) * 0.25
c = ((opp_passTD_allowed/opp_passAtt_allowed)) * 20
d = 2.375 - ((opp_int/opp_passAtt_allowed) * 25)
opp_passerRate_allowed = (((a+b+c+d)/6) *100)
opp_sacks = (18)/games_played
opp_ya = (opp_passYards_allowed/opp_passAtt_allowed)

#Qb vs Opp Defense Data
qb_gamesPlayed_opp = 3

qb_passYards_opp = (632)/qb_gamesPlayed_opp
qb_cmp_opp = (53)/qb_gamesPlayed_opp
qb_att_opp = (81)/qb_gamesPlayed_opp
qb_cmpPer_opp = 65.43
qb_passTD_opp = (6)/qb_gamesPlayed_opp
qb_int_opp = (3)/qb_gamesPlayed_opp
qb_passRate = 98.4
qb_sacked_opp = (5)/games_played
qb_ya_opp = 7.8

# QB Stats Normalized
age = 31.25
completions = (qb_cmp_opp/opp_cmp_allowed) * qb_cmp_opp
attempts = (qb_att_opp/opp_passAtt_allowed) * qb_att_opp
completionPercentage = (qb_cmpPer_opp/opp_cmpPer_allowed) * qb_cmpPer_opp
passTD = (qb_passTD_opp/opp_passTD_allowed) * qb_passTD_opp
interceptions = (qb_int_opp/opp_int) * qb_int_opp
passerRating = (qb_passRate/opp_passerRate_allowed) * qb_passRate
sacksAllowed = (qb_sacked_opp/qb_sacked_opp) * qb_sacked_opp
yardsPerAttempt = (qb_ya_opp/opp_ya) * qb_ya_opp

# Predicted inputs
inputs = [age, completions, attempts, completionPercentage, passTD, interceptions, passerRating, sacksAllowed, yardsPerAttempt]
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