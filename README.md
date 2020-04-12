# ScalaNet

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

## Tensor
1. Add native resource cleaning when GC destroys the reference 
   (use phantom reference + cleanup thread)

## Tensor Flow Low Level
1. Enhance Session
2. Add basics ops
   - move into separate package (`ops`, `ops.math`, etc.)
   - add ability to call operators on `Op`
     (instead of `plus(op1, op2)` should be `op1.plus(op2)` or `op1 + op2`) 
3. Handle tensor graphs
   - eval multiple outputs at the same time 
     (eval on tuples `val (tensor1, tensor2) = (op1, op2).eval`)
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


