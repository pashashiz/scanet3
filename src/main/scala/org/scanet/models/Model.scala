package org.scanet.models

import org.scanet.core.Slice.syntax.::
import org.scanet.core._
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class Model[E: Numeric : TensorType, W: Numeric : TensorType, J: Numeric : TensorType]() extends Serializable {

  def buildResult(x: Output[E], weights: Output[W]): Output[J]

  def result: TF2[E, W, Output[J], Tensor[J]] = TF2(buildResult).returns[Tensor[J]]

  /** @param features number of features in a dataset
    * @return shape of weights tensor
    */
  def weightsShape(features: Int): Shape

  /** @return number of model outputs
   */
  def outputs(): Int

  def buildLoss(x: Output[E], y: Output[E], weights: Output[W]): Output[J]

  def loss: TF3[E, E, W, Output[J], Tensor[J]] = TF3(buildLoss).returns[Tensor[J]]

  def weightsAndGrad: TF3[E, E, W, (Output[W], Output[Float]), (Tensor[W], Tensor[Float])] =
    TF3((x: Output[E], y: Output[E], w: Output[W]) => (w, buildLoss(x, y, w).grad(w))).returns[(Tensor[W], Tensor[Float])]

  def grad: TF3[E, E, W, Output[Float], Tensor[Float]] =
    TF3((x: Output[E], y: Output[E], w: Output[W]) => buildLoss(x, y, w).grad(w)).returns[Tensor[Float]]

  def trained(weights: Tensor[W]) = new TrainedModel(this, weights)

  def splitXY(batch: Output[Float]): (Output[Float], Output[Float]) = {
    // x: (n, m - 1), y: (n, 1)
    val features = batch.shape.dims(1) - outputs()
    val x = batch.slice(::, s2 = 0 until features)
    val y = batch.slice(::, features)
    (x, y)
  }

  def withBias(x: Output[Float]): Output[Float] = {
    val rows = x.shape.dims.head
    Tensor.ones[Float](rows, 1).const.joinAlong(x, 1)
  }

  override def toString: String = getClass.getSimpleName
}