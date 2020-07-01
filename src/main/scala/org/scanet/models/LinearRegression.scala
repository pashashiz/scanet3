package org.scanet.models

import org.scanet.core.{Output, Shape, TensorType}
import org.scanet.math.{Numeric, Floating}
import org.scanet.math.syntax._

/** Ordinary least squares Linear Regression.
  *
  * LinearRegression fits a linear model with coefficients `w = (w1, …, wn)`
  * to minimize the residual sum of squares between the observed targets in the dataset,
  * and the targets predicted by the linear approximation.
  *
  * Model always has only one output
  */
case class LinearRegression[E: Floating: Numeric: TensorType]() extends Model[E, E, E] {

  override def buildResult(x: Output[E], weights: Output[E]): Output[E] = {
    withBias(x) * reshape(weights).transpose
  }

  override def buildLoss(x: Output[E], y: Output[E], weights: Output[E]): Output[E] = {
    val rows = x.shape.dims.head
    (0.5f / rows).const.cast[E] :* (withBias(x) * reshape(weights).transpose - y).pow(2).sum
  }

  private def reshape(weights: Output[E]): Output[E] =
    weights.reshape(1, weights.shape.head)

  override def weightsShape(features: Int): Shape = Shape(features + 1)

  override def outputs(): Int = 1
}
