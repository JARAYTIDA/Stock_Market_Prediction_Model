import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('GOOG.csv', date_parser = True)
data.tail()

data_training = data[data['Date']<'2022-06-01'].copy()
data_test = data[data['Date']>='2022-06-01'].copy()

data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)

scaler = MinMaxScaler()
training = scaler.fit_transform(data_training)
training

X_train = []
y_train = []

for i in range(60, training.shape[0]):
    X_train.append(training[i-60:i])
    y_train.append(training[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

X_train.shape

# BUILDING LSTM MODEL

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

regressor = Sequential()

regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 80, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 120, activation = 'relu'))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

regressor.summary()

regressor.compile(optimizer='adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs=20, batch_size=32)


# PREPARING DATASET

data_test.head()

past_60_days = data_training.tail(60)

df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

inputs = scaler.transform(df)
inputs

X_test = []
y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape

y_pred = regressor.predict(X_test)

scaler.scale_
scale = 1/6.69375383e-03
scale

y_pred = y_pred*scale
y_test = y_test*scale

# calculating error
error = 0
length = len(y_pred)

for i in range(length):
    error = abs(y_test[i] - y_pred[i])/(y_test[i])

error = error/length
error = error*100*scale
error

# claculation of f1-score
from sklearn.metrics import f1_score

def calculate_f1_score(actual_prices, predicted_prices):
    # Convert the price arrays into binary labels (1: increase, 0: decrease)
    actual_labels = [1 if actual_prices[i+1] > actual_prices[i] else 0 for i in range(len(actual_prices) - 1)]
    predicted_labels = [1 if predicted_prices[i+1] > predicted_prices[i] else 0 for i in range(len(predicted_prices) - 1)]
    
    # Calculate the F1-score
    f1 = f1_score(actual_labels, predicted_labels)
    return f1

f1_score = calculate_f1_score(y_test, y_pred)
print("F1-score:", f1_score)

# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
