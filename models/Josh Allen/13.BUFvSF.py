import sys
sys.path.append('../')  # Adjust this path based on where you run the script
import inputs
import model
import pandas as pd

# Josh Allen:
# https://www.pro-football-reference.com/players/A/AlleJo02/gamelog/
# https://www.pro-football-reference.com/players/A/AlleJo02/gamelog/2024/
QB = inputs.Stats(28.195, (217/11), (339/11), (2543/11), (18/11), (5/11), (13/11))

# SF Defense: https://www.pro-football-reference.com/teams/sfo/2024.htm#all_defense
DEF = inputs.Stats(None, (217/11), (344/11), (2128/11), (13/11), (11/11), (28/11))

path = 'JoshAllen.csv'
data = pd.read_csv(path).dropna()
features = ['Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'Int', 'Rate', 'Sk']

alpha = 0.001
ratio = 0.5

# Calc current season last 3 games moving average
ma3s = []
for i in range(0, len(features)):
    s1 = data[features[i]].iloc[-1]
    s2 = data[features[i]].iloc[-2]
    s3 = data[features[i]].iloc[-3]
    ma3s.append(float(s1+s2+s3)/3)

# Normalized Inputs for Model
completions = inputs.normalize(DEF.cmp, QB.cmp, ma3s[0])
attempts = inputs.normalize(DEF.att, QB.att, ma3s[1])
completionsPercentage = (completions/attempts) * 100
passingYards = inputs.normalize(DEF.yards, QB.yards, ma3s[3])
passingTDs = inputs.normalize(DEF.td, QB.td, ma3s[4])
interceptions = ma3s[5]
passerRating = (((((completions/attempts)-0.3)*5) + (((passingYards/attempts)-3)*0.25) + ((passingTDs/attempts)*20) + (2.375-((interceptions/attempts)*25)))/6) *100
sacks = inputs.normalize(DEF.sacks, QB.sacks, ma3s[7])

# Model Inputs
features.remove('Yds')
target = ['Yds']
input_stats = [completions, attempts, completionsPercentage, passingTDs, interceptions, passerRating, sacks]
input_stats = pd.DataFrame([input_stats], columns=features)

prediction = model.run_prediction(data, input_stats, features, target, alpha, ratio)

# Outputs
print(input_stats)
print(f'Defense Average Passing Yards Allowed: {DEF.yards:.2f}')
print(f'Quarter Back Average Passing Yards: {QB.yards:.2f}')
print(f'Last 3 Games Passing Yards Average: {ma3s[3]:.2f}')
print(f'Predicted Passing Yards: {prediction:.2f}')