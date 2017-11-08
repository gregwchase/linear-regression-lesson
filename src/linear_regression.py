from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

'''Preprocessing'''

# Load dataset
boston = load_boston()

# Create the target, and convert to Pandas DataFrame
y = boston.target
boston = pd.DataFrame(boston.data)


# View the Pandas DataFrame
print(boston.head())


# Change the column names; print out the DataFrame again
boston.columns = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']

print(boston.head())

'''Building The Linear Regression Model'''

# Split the data into test and training data sets
X_train, X_test, y_train, y_test = train_test_split(boston, y, test_size=0.20, random_state=42)


# Create the Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)


# Predict against the model
predictions = model.predict(X_test)


# Evaluate the model
print("R2 Score: ", round(r2_score(y_test, predictions),2))
print("Mean Squared Error: ", round(mean_squared_error(y_test, predictions),2))
