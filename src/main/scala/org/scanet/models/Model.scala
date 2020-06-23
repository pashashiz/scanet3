package org.scanet.models

import org.scanet.core.{Output, Shape, TF2, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

case class Model[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType](
   builder: (Output[X], Output[W]) => Output[J],
   shape: Int => Shape) {

  def result: TF2[X, W, Output[J], Tensor[J]] = TF2(builder).returns[Tensor[J]]

  def grad: TF2[X, W, (Output[W], Output[Float]), (Tensor[W], Tensor[Float])] =
    TF2((x: Output[X], w: Output[W]) => (w, builder(x, w).grad(w))).returns[(Tensor[W], Tensor[Float])]

  def display(dir: String = "", x: Shape = Shape(1, 3), w: Shape = Shape(3)): Unit = {
    val (_, _, resultOut) = result.builder(x, w)
    val (_, _, (_, gradOut)) = grad.builder(x, w)
    (resultOut.as("result"), gradOut.as("grad")).display(dir)
  }
}