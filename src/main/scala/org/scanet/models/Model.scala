package org.scanet.models

import org.scanet.core.{Output, Shape, TF2, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

case class Model[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType]
  (builder: (Output[X], Output[W]) => Output[J]) {

  def result: TF2[X, W, Output[J], Tensor[J]] = TF2(builder).returns[Tensor[J]]

  def grad: TF2[X, W, (Output[W], Output[Float]), (Tensor[W], Tensor[Float])] =
    TF2((x: Output[X], w: Output[W]) => (w, builder(x, w).grad(w))).returns[(Tensor[W], Tensor[Float])]

}