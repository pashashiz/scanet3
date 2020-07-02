package org.scanet.models

import org.scanet.core.Slice.syntax.::
import org.scanet.core._
import org.scanet.math.Floating
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class Model[
  E: Numeric: TensorType,
  R: Numeric: Floating: TensorType]() extends Serializable {

  def buildResult(x: Output[E], weights: Output[R]): Output[R]

  def result: TF2[E, R, Output[R], Tensor[R]] = TF2(buildResult).returns[Tensor[R]]

  /** @param features number of features in a edataset
   * @return shape of weights tensor
   */
  def weightsShape(features: Int): Shape

  /** @return number of model outputs
   */
  def outputs(): Int

  def buildLoss(x: Output[E], y: Output[E], weights: Output[R]): Output[R]

  def loss: TF3[E, E, R, Output[R], Tensor[R]] = TF3(buildLoss).returns[Tensor[R]]

  def weightsAndGrad: TF3[E, E, R, (Output[R], Output[R]), (Tensor[R], Tensor[R])] =
    TF3((x: Output[E], y: Output[E], w: Output[R]) => (w, buildLoss(x, y, w).grad(w).returns[R])).returns[(Tensor[R], Tensor[R])]

  def grad: TF3[E, E, R, Output[R], Tensor[R]] =
    TF3((x: Output[E], y: Output[E], w: Output[R]) => buildLoss(x, y, w).grad(w).returns[R]).returns[Tensor[R]]

  def trained(weights: Tensor[R]) = new TrainedModel(this, weights)

  def splitXY(batch: Output[E]): (Output[E], Output[E]) = {
    // x: (n, m - 1), y: (n, 1)
    val features = batch.shape.dims(1) - outputs()
    val x = batch.slice(::, s2 = 0 until features)
    val y = batch.slice(::, features)
    (x, y)
  }

  def withBias(x: Output[E]): Output[E] = {
    val rows = x.shape.dims.head
    Tensor.ones[E](rows, 1).const.joinAlong(x, 1)
  }

  override def toString: String = s"${getClass.getSimpleName}[${TensorType[E].classTag}, ${TensorType[R].classTag}]"
}