# see https://machinelearningmastery.com/understanding-simple-recurrent-neural-networks-in-keras

from pandas import read_csv
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

import matplotlib.pyplot as plt


def create_RNN(hidden_units, dense_units, input_shape, activation):
    # wx weights = (features x units) = (1, 2)
    # wh weights = (units x units) = (2, 2)
    # bh weights = (units) = (2)
    # wy weights = (outputs, units) = (2, 1)
    # by weights = (units) = (1)
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, activation=activation[0], unroll=True))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# input is (time_steps x features) = (3 x 1)
# RNN layer 2 units
# Dense layer 1 unit
test_model = create_RNN(2, 1, (3, 1), activation=['linear', 'linear'])

# let's look at the weights
weights = test_model.get_weights()
wx = weights[0]
wh = weights[1]
bh = weights[2]
wy = weights[3]
by = weights[4]
print('wx = ', wx)
print('wh = ', wh)
print('bh = ', bh)
print('wy =', wy)
print('by = ', by)
print(test_model.summary())

# let's try to compute forward pass manually
# here is an input with 3 time steps
x = np.array([1, 2, 3])
# reshape the input to the required (sample_size x time_steps x features)
x_input = np.reshape(x, (1, 3, 1))
y_pred_model = test_model.predict(x_input)

h0 = np.zeros(2)
h1 = np.dot(x[0], wx) + h0 + bh
h2 = np.dot(x[1], wx) + np.dot(h1, wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2, wh) + bh
o3 = np.dot(h3, wy) + by

print('h1 = ', h1, 'h2 = ', h2, 'h3 = ', h3)

print("Prediction from network ", y_pred_model)
print("Prediction from our computation ", o3)


# we will train the model on time series of sunspots
def get_train_test(url, split_percent=0.8):
    # take only second column (Sunspots)
    df = read_csv(url, usecols=[1], engine='python')
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    print(data)
    n = len(data)
    # Point for splitting data into train and test
    split = int(n * split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    return train_data, test_data, data


sunspots_url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv'
train_data, test_data, data = get_train_test(sunspots_url)


# prepare the input X and target Y
def get_XY(dat, time_steps):
    # Indices of target array
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    # print(Y_ind)
    Y = dat[Y_ind]
    # Prepare X
    rows_x = len(Y)
    X = dat[range(time_steps * rows_x)]
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y


# X = (187, 12, 1) - 187 samples with 12 timeseries elements
# Y = (187) - 187 samples which includes 1st element of next time series (prediction)
time_steps = 12
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

# create real RNN model with tanh
model = create_RNN(hidden_units=3, dense_units=1, input_shape=(time_steps, 1),
                   activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2)


def print_error(trainY, testY, train_predict, test_predict):
    # Error of predictions
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    train_r2_score = r2_score(trainY, train_predict)
    test_r2_score = r2_score(testY, test_predict)
    # Print RMSE/R2 Score
    print('Train RMSE: %.3f RMSE' % (train_rmse))
    print('Test RMSE: %.3f RMSE' % (test_rmse))
    print('Train R2 Score: %.3f RMSE' % (train_r2_score))
    print('Test R2 Score: %.3f RMSE' % (test_r2_score))


# make 187 predictions (one prediction for each year)
train_predict = model.predict(trainX)
test_predict = model.predict(testX)
# mean square error
print_error(trainY, testY, train_predict, test_predict)


# plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    plt.savefig('sunspots_forecast.png')


plot_result(trainY, testY, train_predict, test_predict)
