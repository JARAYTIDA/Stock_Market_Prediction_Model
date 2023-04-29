import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
# %matplotlib inline
import numpy as np

dataset = pd.read_csv('GOOG.csv')
dataset.head()

dataset['Date'] = pd.to_datetime(dataset.Date)
dataset.shape
dataset.drop('Adj Close', axis=1, inplace=True)
dataset.head()
dataset.isnull().sum()
dataset.isna().any()
dataset.info()
dataset.describe()
print(len(dataset))
dataset['Open'].plot(figsize=(16,6))


x = dataset[['Open', 'High', 'Low', 'Volume']]
y = dataset['Close']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,random_state=0)

X_train.shape
X_test.shape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regressor = LinearRegression()

regressor.fit(X_train,Y_train)
print(regressor.coef_)
predicated = regressor.predict(X_test)
dframe = pd.DataFrame(Y_test, predicated)

dfr = pd.DataFrame({'Actual Price':Y_test, 'Predicated Price':predicated})
dfr
regressor.score(X_test, Y_test)


import math
print("Mean Absolute Error: ", metrics.mean_absolute_error(Y_test, predicated) )
print("Mean Square Error: ", metrics.mean_squared_error(Y_test, predicated) )
print("Root Mean Square Error: ", math.sqrt(metrics.mean_squared_error(Y_test, predicated)) )

X = np.array(dfr)
## error calcualtion

error = 0
length = len(X)

for i in range(length):
    error = abs(X[i][0]*1000 - X[i][1]*1000)/(X[i][0])

error = error/length
error = error*100
error

from sklearn.metrics import f1_score

def calculate_f1_score(actual_prices, predicted_prices):
    # Convert the price arrays into binary labels (1: increase, 0: decrease)
    actual_labels = [1 if actual_prices[i+1] > actual_prices[i] else 0 for i in range(len(actual_prices) - 1)]
    predicted_labels = [1 if predicted_prices[i+1] > predicted_prices[i] else 0 for i in range(len(predicted_prices) - 1)]
    
    # Calculate the F1-score
    f1 = f1_score(actual_labels, predicted_labels)
    return f1

Y = []
x = []
for i in range(length):
    Y.append(X[i][1])
    x.append(X[i][0])

f1_score = calculate_f1_score(x, Y)
print("F1-score:", f1_score)

graph = dfr.head(10)
graph.plot(kind='bar')