from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

x, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
print(x[0:5])
print(y[0:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(1, input_shape=(2,), activation='sigmoid'))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, epochs=200, verbose=1)

loss, accuracy = model.evaluate(x_test, y_test)
print("Loss: {}, accuracy: {}".format(loss, accuracy))


