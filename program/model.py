import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, mean_squared_error


# Split the data
def split_data(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=69)
    return X_train, X_test, y_train, y_test

# Train the model
def make_model(X_train, y_train):
    model = Lasso(alpha=1)
    model.fit(X_train, y_train)
    return model

# Test the model
def test_model(model, X_test, y_test):
    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)
    maePer = mean_absolute_percentage_error(y_test, prediction) * 100
    mse = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)
    print(f'Mean Absolute Error: {mae:.5f}\nMAE%: {maePer:.2f} %\nMean Squared Error: {mse:.5f}\nR2 Score: {r2:.5f}')


def run_prediction(data, inputs, features, target):

    X_train, X_test, y_train, y_test = split_data(data[features], data[target])
    model = make_model(X_train, y_train)
    test_model(model, X_test, y_test)

    # Convert prediction result to float
    prediction = model.predict(inputs)
    prediction = prediction[0]
    prediction = float(prediction)

    return prediction