package org.scanet.models

import org.scanet.core._
import org.scanet.math.Floating
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class Model extends Serializable {

  def build[E: Numeric: Floating: TensorType](x: Output[E], weights: Output[E]): Output[E]

  def result[E: Numeric: Floating: TensorType]: TF2[E, E, Output[E], Tensor[E]] = TF2(build[E]).returns[Tensor[E]]

  /** @param features number of features in a edataset
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

  def loss[E: Numeric: Floating: TensorType]: TF3[E, E, E, Output[E], Tensor[E]] = TF3(
    (x: Output[E], y: Output[E], weights: Output[E]) => build[E](x, y, weights)).returns[Tensor[E]]

  def weightsAndGrad[E: Numeric: Floating: TensorType]: TF3[E, E, E, (Output[E], Output[E]), (Tensor[E], Tensor[E])] =
    TF3((x: Output[E], y: Output[E], w: Output[E]) => (w, build(x, y, w).grad(w).returns[E])).returns[(Tensor[E], Tensor[E])]

  def grad[E: Numeric: Floating: TensorType]: TF3[E, E, E, Output[E], Tensor[E]] =
    TF3((x: Output[E], y: Output[E], w: Output[E]) => build(x, y, w).grad(w).returns[E]).returns[Tensor[E]]

  def trained[E: Numeric: Floating: TensorType](weights: Tensor[E]) = new TrainedModel(this, weights)

  override def toString: String = s"$model:$lossF"
}

class TrainedModel[E: Numeric : Floating: TensorType](val lossModel: LossModel, val weights: Tensor[E]) {

  def buildResult(x: Output[E]): Output[E] = lossModel.model.build(x, weights.const)

  def result: TF1[E, Output[E], Tensor[E]] =
    TF1((x: Output[E]) => buildResult(x)).returns[Tensor[E]]

  def buildLoss(x: Output[E], y: Output[E]): Output[E] = lossModel.build(x, y, weights.const)

  def loss: TF2[E, E, Output[E], Tensor[E]] =
    TF2((x: Output[E], y: Output[E]) => buildLoss(x, y)).returns[Tensor[E]]

  def outputs(): Int = lossModel.model.outputs()
}
