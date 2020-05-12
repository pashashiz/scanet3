# ScalaNet

[![Build Status](https://travis-ci.org/pashashiz/scanet3.svg?branch=master)](https://travis-ci.org/pashashiz/scanet3)


Type-safe, high performance, distributed Neural networks in Scala (not Python, finally...).

## Architecture Intro

Low level (linear algebra) operations powered by low level TensorFlow API (C, C++ bindings via JNI). 

Scala used to build computation graphs and compile them into native tensor graphs.
Compiled graphs are fully calculated in native code (on CPU, GPU or TPU) 
and only result is returned back via `DirectBuffer`which points into native memory. 

`DirectBuffer` is wrapped with `Tensor` object which allows 
to slice and read data in a convenient way (just like `Breeze` or `Numpy` does).

todo

## Road Map

## Tensor Flow Low Level
That is done for now

## Optimizers
1. Add SGD, AdaGrad + AdaDelta, RMSProp, Adam + Nadam
2. Minimize basic math functions
3. Add logging
4. Add more stop conditions (check which are used right now in Keras)
5. Minimize Linear regression + r2 score estimator
6. Minimize Logistic regression + accuracy estimator, confusion matrix, precision, recall, f1 score
7. Add runtime tensorboard chart
8. Investigate the way to parallelize optimizer
9. Check the way to read data into dataset


## CPU & GPU & TPU banchmarks
1. Create computation intensive operation, like `matmul` multiple times large tensors
   and compare with Scala `breeze`, python `tensorflow`, python `numpy`
2. Improve performance if needed

## NN baseline + Multilayer Perceptron NN + Evaluation-2
1. todo

## Additional Layers (dropout, etc.)
1. todo

## Convolutional NN
1. todo

## Recurent NN
1. todo

## Distributed Processing (Spark)
1. todo


