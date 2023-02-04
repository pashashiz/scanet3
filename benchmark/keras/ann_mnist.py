import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = (x_train.astype("float32") / 255).reshape((-1, 784))
x_test = (x_test.astype("float32") / 255).reshape((-1, 784))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential([
  Dense(50, activation='sigmoid', input_shape=(784,)),
  Dense(10, activation='softmax')
])

model.compile(
  optimizer=keras.optimizers.Adam(lr=0.002),
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

start = time.time()
# Train the model.
model.fit(
  x_train,
  y_train,
  epochs=200,
  batch_size=1000,
)
end = time.time()
print("took (sec):", end - start)