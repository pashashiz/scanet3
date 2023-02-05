import pandas as pd

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.metrics import r2_score

df = pd.read_csv('data/weight-height.csv')
print(df.head())

x = df[['Height']].values
y = df[['Weight']].values

model = Sequential()
model.add(Dense(1, input_shape=(1,)))
model.compile(optimizer=Adam(lr=0.5), loss='mean_squared_error')
print(model.summary())

model.fit(x, y, epochs=30)

y_pred = model.predict(x)
print(y[0:5])
print(y_pred[0:5])

print("R2 score is {:0.3f}".format(r2_score(y, y_pred)))