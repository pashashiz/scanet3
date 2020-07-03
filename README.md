# ScalaNet

[![Build Status](https://travis-ci.org/pashashiz/scanet3.svg?branch=master)](https://travis-ci.org/pashashiz/scanet3)


Type-safe, high performance, distributed Neural networks in Scala (not Python, finally...).

## Intro

Low level (linear algebra) operations powered by low level TensorFlow API (C, C++ bindings via JNI). 

Scala used to build computation graphs and compile them into native tensor graphs.
Compiled graphs are fully calculated in native code (on CPU, GPU or TPU) 
and only result is returned back via `DirectBuffer`which points into native memory. 

`DirectBuffer` is wrapped with `Tensor` object which allows 
to slice and read data in a convenient way (just like `Breeze` or `Numpy` does).

The optimizer is built on `Spark` and can optimize the model in a distributed/parallel way.
The input data is expected to be `RDD[Array[TensorType]`. 
Usually, `TensorType` is choosen to be  `Float` since it performs best on GPU, but that is not
limited to it.

Example of solving a linear regression model:

``` scala
val ds = facebookComments // RDD[Array[Float]]
val trained = ds.train(LinearRegression)
  .loss(MeanSquaredError)
  .using(Adam(rate = 0.1f))
  .batch(1000)
  .each(1.epochs, logResult())
  .each(1.iterations, plotResult(name = "Error", dir = "board/Adam"))
  .stopAfter(10.epochs)
  .run()
val loss = trained.loss.compile()
val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
loss(x, y).toScalar should be <= 1f
```

Here, `loss` will be logged as well as added to `TensorBoard`. 
To check live optimization process you can:
```sh
tensorboard --logdir board
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

### Models
- [x] Linear Regression
- [ ] Simple math models for benchmarks
- [x] Binary Logistic Regression
- [ ] ANN (Multilayer Perceptron NN)
- [ ] Layers Dropout, Regularization, Normalization
- [ ] Convolutional NN
- [ ] Recurent NN
- [ ] others

### Preprocessing
- [ ] Auto-converting into `RDD[Array[TensorType]]`
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