package org.scanet.core

import org.tensorflow.{Tensor => NativeTensor}

import scala.{specialized => sp}

// Op[A] => Tensor[A]
// (Op[A], Op[B]) => (Tensor[A], Tensor[B])
// (Op[A], Op[B], Op[C]) => ...

// todo: see how shapeless handles that, for now use simple way without typeclasses

case class OutputEval[@sp A: TensorType](out: Output[A]) {
  def eval: Tensor[A] = {
    Session.run(out)
  }
  def display(dir: String = ""): Unit = {
    TensorBoard.write(List(out), dir)
  }
}

case class Tuple2Eval[@sp A1: TensorType, @sp A2: TensorType](tuple: (Output[A1], Output[A2])) {
  def eval: (Tensor[A1], Tensor[A2]) = {
    val tensors = Session.runN(List(tuple._1, tuple._2))
    (Tensor[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
      Tensor[A2](tensors(1).asInstanceOf[NativeTensor[A2]]))
  }
  def display(dir: String = ""): Unit = {
    TensorBoard.write(List(tuple._1, tuple._2), dir)
  }
}

case class Tuple3Eval[A1: TensorType, A2: TensorType, A3: TensorType](tuple: (Output[A1], Output[A2], Output[A3])) {
  def eval: (Tensor[A1], Tensor[A2], Tensor[A3]) = {
    val tensors = Session.runN(List(tuple._1, tuple._2, tuple._3))
    (Tensor[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
      Tensor[A2](tensors(1).asInstanceOf[NativeTensor[A2]]),
      Tensor[A3](tensors(2).asInstanceOf[NativeTensor[A3]])
    )
  }
  def display(dir: String = ""): Unit = {
    TensorBoard.write(List(tuple._1, tuple._2, tuple._3), dir)
  }
}

object Eval {
  trait Syntax {
    implicit def outputEval[A: TensorType](out: Output[A]): OutputEval[A] = OutputEval(out)
    implicit def tuple2Eval[A1: TensorType, A2: TensorType](tuple: (Output[A1], Output[A2])): Tuple2Eval[A1, A2] = Tuple2Eval(tuple)
    implicit def tuple3Eval[A1: TensorType, A2: TensorType, A3: TensorType](tuple: (Output[A1], Output[A2], Output[A3])): Tuple3Eval[A1, A2, A3] = Tuple3Eval(tuple)
  }
  object syntax extends Syntax
}
