# Scanet

[![Build Status](https://travis-ci.org/pashashiz/scanet3.svg?branch=master)](https://travis-ci.org/pashashiz/scanet3)


Type-safe, high performance, distributed Neural networks in Scala (not Python, finally...).

## Intro

Low level (linear algebra) operations powered by low level TensorFlow API (C, C++ bindings via JNI). 

Scala used to build computation graphs and compile them into native tensor graphs.
Compiled graphs are fully calculated in native code (on CPU, GPU or TPU) 
and only result is returned back via `DirectBuffer`which points into native memory. 

`DirectBuffer` is wrapped with `Tensor` read-only object which allows 
to slice and read data in a convenient way (just like `Breeze` or `Numpy` does).

The optimizer is built on top of `Spark` and can optimize the model in a distributed/parallel way.
The chosen algorithm - `Data parallelism with synchronous model averaging`. The dataset is split between
the workers and each epoch is run independently on each data split, at the end of each epoch
parameters are averaged and broadcasted back to each worker.

The input data is expected to be `Dataset[Array[TensorType]` and it contains a shape of the tensors in metadata.
Usually, `TensorType` is choosen to be `Float` since it performs best on GPU, also `Double` can be used.

## Examples

### ANN

Example of a simple MNIST dataset classifier with Fully Connected Neural Network:

``` scala
val (trainingDs, testDs) = MNIST.load(sc, trainingSize = 30000)
val model = Dense(50, Sigmoid) >> Dense(10, Softmax)
val trained = trainingDs.train(model)
  .loss(CategoricalCrossentropy)
  .using(Adam(0.01f))
  .batch(1000)
  .each(1.epochs, RecordLoss(tensorboard = true))
  .each(10.epochs, RecordAccuracy(testDs, tensorboard = true))
  .stopAfter(200.epochs)
  .run()
accuracy(trained, testDs) should be >= 0.95f
```

Here, `loss` and `accuracy` will be logged and added to `TensorBoard` as live trends. To run tensorboard execute:
```sh
pip install tensorboard
tensorboard --logdir board
```

### CNN

Same but with CNN (Convolutional Neural Network)
```scala
val (trainingDs, testDs) = MNIST()
val model =
  Conv2D(32, activation = ReLU()) >> Pool2D() >>
  Conv2D(64, activation = ReLU()) >> Pool2D() >>
  Flatten >> Dense(10, Softmax)
val trained = trainingDs
  .train(model)
  .loss(CategoricalCrossentropy)
  .using(Adam(0.001f))
  .batch(100)
  .each(1.epochs, RecordLoss(tensorboard = true))
  .each(1.epochs, RecordAccuracy(testDs, tensorboard = true))
  .stopAfter(3.epochs)
  .run()
accuracy(trained, testDs) should be >= 0.98f
```
 
## Road Map

### Tensor Flow Low Level API
- [x] Tensor
- [x] DSL for computation DAG 
- [x] TF Session
- [x] Core ops
- [x] Math ops
- [x] Logical ops
- [x] String ops
- [x] TF Functions, Placeholders, Session caching
- [x] Tensor Board basic support

### Optimizer engine
- [x] Spark
- [ ] Hyper parameter tuning
- [ ] Model Import/Export
 
### Optimizer algorithms
- [x] SGD
- [x] AdaGrad
- [x] AdaDelta
- [x] RMSProp
- [x] Adam
- [x] Nadam
- [x] Adamax
- [x] AMSGrad

### Statistics
- [x] Variance/STD
- [ ] Covariance/Correlation Matrix
- [ ] Lots of other useful algs to analize the data set

### Models
- [x] Linear Regression
- [ ] Simple math models for benchmarks
- [x] Binary Logistic Regression
- [x] ANN (Multilayer Perceptron NN)
- [x] kernel regularization
- [ ] Layers Dropout, Batch Normalization
- [x] Convolutional NN
- [ ] Recurrent NN
- [ ] others

### Activation functions
- [x] Sigmoid
- [x] Tanh
- [x] RELU
- [x] Softmax
- [ ] Exp
- [ ] SELU
- [ ] ELU
- [ ] Sofplus

### Loss functions
- [x] RMSE (Mean Squared Error)
- [x] Binary Crossentropy
- [x] Categorical Crossentropy

### Benchmark Datasets
- [x] MNIST

### Preprocessing
- [ ] Feature scalers
- [ ] Feature embedding
- [ ] Hashed features
- [ ] Crossed features

### Estimators
- [ ] r2 score
- [x] accuracy estimator, 
- [ ] confusion matrix, precision, recall, f1 score
- [ ] runtime estimating and new stop condition based on that

### CPU & GPU & TPU banchmarks
- [ ] Create computation intensive operation, like `matmul` multiple times large tensors
      and compare with Scala `breeze`, python `tensorflow`, python `numpy`
- [ ] Compare with existing implementations using local CPU
- [ ] Compare with existing implementations using one GPU
- [ ] Compare with existing implementations using distributed mode on GCP DataProc

### Other useful things
- [ ] While training analyze the weights histograms to make sure the deep NN do not saturate
- [ ] Grid/Random hyper parameters search
- [x] Different weight initializers (Xavier)
- [ ] Decay learning rate over time (step, exponential, 1/t decay)
- [ ] Try using in interactive notebook
- [ ] Add graph library so we could plot some charts and publish them in `tensorboard` or `notebook` (maybe fork and upgrade `vegas` to scala `2.12` ot try `evil-plot`)

### Refactoring
- Add DSL to build tensor requirements like `tensor require rank(4)`, `tensor require shape squratedMatrix`

If you want to become a contributor, you are welcome!!! You can pick anything from a Road Map or propose your idea. 
 
 Please, contact:
- `ppoh@softserveinc.com`
- `yatam@softserveinc.com`