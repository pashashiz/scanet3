package org.scanet.models

import org.scanet.core.{Output, TF1, TF2, Tensor, TensorType}
import org.scanet.math.{Numeric, Floating}
import org.scanet.math.syntax._

class TrainedModel[
  E: Numeric : TensorType,
  R: Numeric : Floating : TensorType](
     val model: Model[E, R], val weights: Tensor[R]) {

  def buildResult(x: Output[E]): Output[R] = model.buildResult(x, weights.const)

  def result: TF1[E, Output[R], Tensor[R]] =
    TF1((x: Output[E]) => buildResult(x)).returns[Tensor[R]]

  def buildLoss(x: Output[E], y: Output[E]): Output[R] = model.buildLoss(x, y, weights.const)

  def loss: TF2[E, E, Output[R], Tensor[R]] =
    TF2((x: Output[E], y: Output[E]) => buildLoss(x, y)).returns[Tensor[R]]

  def outputs(): Int = model.outputs()
}
