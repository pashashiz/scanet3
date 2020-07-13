package org.scanet.models

import org.scanet.core._
import org.scanet.math.Floating
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class Model extends Serializable {

  def build[E: Numeric: Floating: TensorType](x: Output[E], weights: Output[E]): Output[E]

  def result[E: Numeric: Floating: TensorType]: TF2[E, Tensor[E], E, Tensor[E], Output[E]] = TF2(build[E])

  /** @param features number of features in a dataset
   * @return shape of weights tensor
   */
  def shape(features: Int): Shape

  /** @return number of model outputs
   */
  def outputs(): Int

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  def withBias[E: Numeric: Floating: TensorType](x: Output[E]): Output[E] = {
    val rows = x.shape.dims.head
    Tensor.ones[E](rows, 1).const.joinAlong(x, 1)
  }
}

case class LossModel(model: Model, lossF: Loss) extends Serializable {

  def build[E: Numeric: Floating: TensorType](x: Output[E], y: Output[E], weights: Output[E]): Output[E] =
    lossF.build(model.build(x, weights), y)

  def loss[E: Numeric: Floating: TensorType]: TF3[E, Tensor[E], E, Tensor[E], E, Tensor[E], Output[E]] =
    TF3(build[E])

  def weightsAndGrad[E: Numeric: Floating: TensorType] =
    TF3[Output, E, Output, E, Output, E, (Output[E], Output[E])](
      (x, y, w) => (w, build(x, y, w).grad(w).returns[E]))

  def grad[E: Numeric: Floating: TensorType] =
    TF3[Output, E, Output, E, Output, E, Output[E]](
      (x, y, w) => build(x, y, w).grad(w).returns[E])

  def trained[E: Numeric: Floating: TensorType](weights: Tensor[E]) = new TrainedModel(this, weights)

  override def toString: String = s"$model:$lossF"
}

class TrainedModel[E: Numeric : Floating: TensorType](val lossModel: LossModel, val weights: Tensor[E]) {

  def buildResult(x: Output[E]): Output[E] = lossModel.model.build(x, weights.const)

  def result: TF1[E, Tensor[E], Output[E]] = TF1(buildResult)

  def buildLoss(x: Output[E], y: Output[E]): Output[E] = lossModel.build(x, y, weights.const)

  def loss: TF2[E, Tensor[E], E, Tensor[E], Output[E]] =
    TF2[Output, E, Output, E, Output[E]]((x, y) => buildLoss(x, y))

  def outputs(): Int = lossModel.model.outputs()
}