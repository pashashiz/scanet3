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

## Tasks

native
core
math
linalg
syntax.{core, math, linalg}

## Tensor Flow Low Level
1. Enhance Session
2. Add basics ops
   - move into separate package (`ops`, `ops.math`, etc.)
   - add ability to call operators on `Op`
     (instead of `plus(op1, op2)` should be `op1.plus(op2)` or `op1 + op2`) 
3. Handle tensor graphs
   - support ops with multiple outputs 
     (use tuples: `val (op1, op2) = op(...)`)
4. Add gradient support to ops 
   (check what standard lib offers and reuse if possible, if not make own)
5. Add the rest of the ops

## CPU & GPU & TPU banchmarks
1. todo

## Distributed Processing (Spark)
1. todo

## Optimizers
1. todo

## NN baseline + Multilayer Perceptron NN + Evaluation
1. todo

## Additional Layers (dropout, etc.)
1. todo

## Convolutional NN
1. todo

## Recurent NN
1. todo


