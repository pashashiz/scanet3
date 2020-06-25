package org.scanet.models

import org.scanet.core.{Output, Shape, TF2, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

case class Model[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType](
   name: String,
   builder: (Output[X], Output[W]) => Output[J],
   shape: Int => Shape) {

  def loss: TF2[X, W, Output[J], Tensor[J]] = TF2(builder).returns[Tensor[J]]

  def weightsAndGrad: TF2[X, W, (Output[W], Output[Float]), (Tensor[W], Tensor[Float])] =
    TF2((x: Output[X], w: Output[W]) => (w, builder(x, w).grad(w))).returns[(Tensor[W], Tensor[Float])]

  def grad: TF2[X, W, Output[Float], Tensor[Float]] =
    TF2((x: Output[X], w: Output[W]) => builder(x, w).grad(w)).returns[Tensor[Float]]

  def display(dir: String = "", x: Shape = Shape(1, 3), w: Shape = Shape(3)): Unit = {
    val (_, _, resultOut) = loss.builder(x, w)
    val (_, _, (_, gradOut)) = weightsAndGrad.builder(x, w)
    (resultOut.as("loss"), gradOut.as("grad")).display(dir)
  }

  override def toString: String = name
}