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
 
### RNN

Simple RNN (Recurrent Neural Network) to forecast sunspots 
```scala
val Array(train, test) = monthlySunspots(12).randomSplit(Array(0.8, 0.2), 1)
val model = RNN(SimpleRNNCell(units = 3)) >> Dense(1, Tanh)
val trained = train
  .train(model)
  .loss(MeanSquaredError)
  .using(Adam())
  .batch(10)~~~~
  .each(1.epochs, RecordLoss(tensorboard = true))
  .stopAfter(100.epochs)
  .run()
RMSE(trained, test) should be < 0.1f
R2Score(trained, test) should be > 0.8f
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
- [ ] Lots of other useful algs to analyze the data set

### Models
- [x] Linear Regression
- [ ] Simple math models for benchmarks
- [x] Binary Logistic Regression
- [x] ANN (Multilayer Perceptron NN)
- [x] kernel regularization
- [ ] Layers Dropout (provide random generator to layers)
- [ ] Batch Normalization
- [x] Convolutional NN
- [ ] Recurrent NN
- [ ] others

# Localization & Object Detection & Instance Segmentation
- [ ] CNN with Localization
- [ ] Region Proposals (Selective Search, EdgeBoxes, etc..)
- [ ] R-CNN
- [ ] Fast R-CNN
- [ ] Faster R-CNN

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
- [ ] Boston Housing price regression dataset
- [x] MNIST
- [ ] Fashion MNIST 
- [ ] CIFAR-10
- [ ] CIFAR-100
- [ ] ILSVRC (ImageNet-1000)

### Preprocessing
- [ ] SVD/PCA/Whitening
- [ ] Feature scalers
- [ ] Feature embedding
- [ ] Hashed features
- [ ] Crossed features

### Estimators
- [ ] r2 score
- [x] accuracy estimator, 
- [ ] confusion matrix, precision, recall, f1 score
- [ ] runtime estimating and new stop condition based on that

### Benchmarks
- [ ] LeNet 
- [ ] AlexNet
- [ ] ZF Net
- [ ] ZF Net
- [ ] VGGNet
- [ ] GoogLeNet
- [ ] ResNet
- [ ] ...

### CPU vs GPU vs TPU 
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
- Redefine the way we train model on a dataset and make a prediction. 
  We should cover 2 cases: BigData with `spark` which can train and predict on large datasets and single (batch) prediction without `spark` dependency (to be able to expose model via API or use in realtime).
  For that we need to:
  + separate project into `core` + `spark` modules.
  + implement model weights export/import
  + implement feature preprocessing, for training use case try using `MLib`, 
    yet we need to figure out how to transform features via regular function without `spark` involved
  + integrating with `MLib` might require redefining the `Dataset[Record[A]]` we have right now
    probably better to use any abstract dataset which contains 2 required columns `features` + `labels`
    for training and `features` for prediction.
- Add DSL to build tensor requirements like `tensor require rank(4)`, `tensor require shape squratedMatrix`

If you want to become a contributor, you are welcome!!! You can pick anything from a Road Map or propose your idea. 
 
 Please, contact:
- `ppoh@softserveinc.com`
- `yatam@softserveinc.com`