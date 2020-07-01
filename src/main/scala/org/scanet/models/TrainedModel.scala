package org.scanet.models

import org.scanet.core.{Output, TF1, TF2, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

class TrainedModel[E: Numeric : TensorType, W: Numeric : TensorType, J: Numeric : TensorType](
    val model: Model[E, W, J], val weights: Tensor[W]) {

  def buildResult(x: Output[E]): Output[J] = model.buildResult(x, weights.const)

  def result: TF1[E, Output[J], Tensor[J]] =
    TF1((x: Output[E]) => buildResult(x)).returns[Tensor[J]]

  def buildLoss(x: Output[E], y: Output[E]): Output[J] = model.buildLoss(x, y, weights.const)

  def loss: TF2[E, E, Output[J], Tensor[J]] =
    TF2((x: Output[E], y: Output[E]) => buildLoss(x, y)).returns[Tensor[J]]

  def outputs(): Int = model.outputs()
}
