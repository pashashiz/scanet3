package org.scanet.models

import org.scanet.core.{Output, Shape}
import org.scanet.math.syntax._

/** Ordinary least squares Linear Regression.
  *
  * LinearRegression fits a linear model with coefficients `w = (w1, …, wn)`
  * to minimize the residual sum of squares between the observed targets in the dataset,
  * and the targets predicted by the linear approximation.
  *
  * Model always has only one output
  */
object LinearRegression extends Model[Float, Float, Float] {

  override def buildResult(x: Output[Float], weights: Output[Float]): Output[Float] = {
    withBias(x) * reshape(weights).transpose
  }

  override def buildLoss(x: Output[Float], y: Output[Float], weights: Output[Float]): Output[Float] = {
    val rows = x.shape.dims.head
    (0.5f / rows).const :* (withBias(x) * reshape(weights).transpose - y).pow(2).sum
  }

  private def reshape(weights: Output[Float]): Output[Float] =
    weights.reshape(1, weights.shape.head)

  override def weightsShape(features: Int): Shape = Shape(features + 1)

  override def outputs(): Int = 1
}
