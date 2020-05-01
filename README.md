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
1. Add logging
2. Add gradient support to ops.
   (Check what standard lib offers and reuse if possible, if not make own.
   Would be nice to make a separate research project and create POC for few ops `[+, *]`)
3. Check if we need mutable `variable` and `feeding`. 
   I guess we would need `variable` so we could save data in native memory, switch to JVM and went back.
4. Check how we can use flow control ops (`if`, `while`, `for`)
5. Add device placement support in an idiomatic way
6. Check if we can add namespaces in a graph in an idiomatic way
7. Add the rest of the math ops (we would need at least `min`, `max`, `sum`, `mean`, `pow`, `sigmoid`)
  (to be compatible with [breeze](https://github.com/scalanlp/breeze/wiki/Linear-Algebra-Cheat-Sheet))
8. Add ability to write charts to tensor-board. We would need that to see how optimizer works.

NOTE: Check if we need something else [here](https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/extend/bindings.md)

## CPU & GPU & TPU banchmarks
1. Create computation intensive operation, like `matmul` multiple times large tensors
   and compare with Scala `breeze`, python `tensorflow`, python `numpy`

## Optimizers
1. Add SGD
2. Add AdaGrad + AdaDelta
4. Add RMSProp
5. Add Adam + Nadam

## Regression
1. Linear regression
2. Logistic regression

## Evaluation-1
1. Implement evaluation 

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


