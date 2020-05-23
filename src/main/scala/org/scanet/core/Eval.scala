package org.scanet.core

import org.scanet.core.Session.using

import scala.{specialized => sp}

case class OutputEval[@sp A: TensorType](out: Output[A]) {
  def eval: Tensor[A] = using(_.runner.eval(out))
  def display(dir: String = ""): Unit = {
    new TensorBoard(dir).addGraph(out)
  }
}

case class Tuple2Eval[@sp A1: TensorType, @sp A2: TensorType](tuple: (Output[A1], Output[A2])) {
  def eval: (Tensor[A1], Tensor[A2]) = using(_.runner.eval(tuple._1, tuple._2))
  def display(dir: String = ""): Unit = {
    new TensorBoard(dir).addGraph(List(tuple._1, tuple._2): _*)
  }
}

case class Tuple3Eval[A1: TensorType, A2: TensorType, A3: TensorType](tuple: (Output[A1], Output[A2], Output[A3])) {
  def eval: (Tensor[A1], Tensor[A2], Tensor[A3]) = using(_.runner.eval(tuple._1, tuple._2, tuple._3))
  def display(dir: String = ""): Unit = {
    new TensorBoard(dir).addGraph(List(tuple._1, tuple._2, tuple._3): _*)
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
