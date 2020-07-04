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
case object LinearRegression extends Model {

  override def build[A: Numeric: Floating: TensorType](x: Output[A], weights: Output[A]): Output[A] =
    withBias(x) * reshape(weights).transpose

  private def reshape[A: TensorType](weights: Output[A]): Output[A] =
    weights.reshape(1, weights.shape.head)

  override def shape(features: Int): Shape = Shape(features + 1)

  override def outputs(): Int = 1
}
