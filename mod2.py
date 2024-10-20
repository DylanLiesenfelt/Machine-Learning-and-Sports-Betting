import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

data = 'DakPrescott.csv'
features = ['Cmp', 'Att','Pass_TD', 'Passer_rating','Y/A']
target = ['Pass_Yds']

inputs = [15.61,22.43,3.00, 98.40, 9.17]
test = pd.DataFrame([inputs], columns=features)


def pass_prediction(data, features, target):
    df = read_data(data)

    X_train, X_test, y_train, y_test = split_data(df[features], df[target])

    model = train_model(X_train, y_train)
    test_model(model, X_test, y_test)

    print(f'Predicted Outcome{model.predict(test)[0]}')

# Read data set
def read_data(data):
    df = pd.read_csv(data)
    df = df.dropna()
    return df

# Split data
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69, shuffle=True)
    return X_train, X_test, y_train, y_test

# Train the model
def train_model(X_train, y_train):
    linear_reg = LinearRegression()
    linear_reg.fit(X_train, y_train)
    return linear_reg

# Tes the model
def test_model(model, X_test, y_test):
    prediction = model.predict(X_test)

    dif = []
    count = 0
    for i in range(len(prediction)):
        dif.append(y_test.values[i]-prediction[i])
        count += 1
        print(f'#: {i} PRED: {prediction[i]} ACTUAL: {y_test.values[i]} DIF: {y_test.values[i]-prediction[i]}')

    print(f'Mean Margin of Error: {sum(dif)/count}')
    print(f'Median Margin of Error')
# Execute
pass_prediction(data, features, target)