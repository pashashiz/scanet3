package org.scanet.core

import java.nio.file.{Path, Paths}

import org.tensorflow.{Tensor => NativeTensor}

import scala.{specialized => sp}

// Op[A] => Tensor[A]
// (Op[A], Op[B]) => (Tensor[A], Tensor[B])
// (Op[A], Op[B], Op[C]) => ...

// todo: see how shapeless handles that, for now use simple way without typeclasses

case class OutputEval[@sp A: TfType](out: Output[A]) {
  def eval: Tensor[A] = {
    Session.run(out)
  }
  def display(dir: Path = Paths.get("")): Unit = {
    TensorBoard.write(List(out), dir)
  }
}

case class Tuple2Eval[@sp A1: TfType, @sp A2: TfType](tuple: (Output[A1], Output[A2])) {
  def eval: (Tensor[A1], Tensor[A2]) = {
    val tensors = Session.runN(List(tuple._1, tuple._2))
    (Tensor[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
      Tensor[A2](tensors(1).asInstanceOf[NativeTensor[A2]]))
  }
  def display(dir: Path = Paths.get("")): Unit = {
    TensorBoard.write(List(tuple._1, tuple._2), dir)
  }
}

case class Tuple3Eval[A1: TfType, A2: TfType, A3: TfType](tuple: (Output[A1], Output[A2], Output[A3])) {
  def eval: (Tensor[A1], Tensor[A2], Tensor[A3]) = {
    val tensors = Session.runN(List(tuple._1, tuple._2, tuple._3))
    (Tensor[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
      Tensor[A2](tensors(1).asInstanceOf[NativeTensor[A2]]),
      Tensor[A3](tensors(2).asInstanceOf[NativeTensor[A3]])
    )
  }
  def display(dir: Path = Paths.get("")): Unit = {
    TensorBoard.write(List(tuple._1, tuple._2, tuple._3), dir)
  }
}

object Eval {
  trait Syntax {
    implicit def outputEval[A: TfType](out: Output[A]): OutputEval[A] = OutputEval(out)
    implicit def tuple2Eval[A1: TfType, A2: TfType](tuple: (Output[A1], Output[A2])): Tuple2Eval[A1, A2] = Tuple2Eval(tuple)
    implicit def tuple3Eval[A1: TfType, A2: TfType, A3: TfType](tuple: (Output[A1], Output[A2], Output[A3])): Tuple3Eval[A1, A2, A3] = Tuple3Eval(tuple)
  }
  object syntax extends Syntax
}
