package scanet.models

import scanet.core.{Expr, OutputSeq, Shape, TensorType}
import scanet.math.{Floating, Numeric}
import scanet.math.syntax._
import scala.collection.immutable.Seq

/** Ordinary least squares Linear Regression.
  *
  * LinearRegression fits a linear model with coefficients `w = (w1, …, wn)`
  * to minimize the residual sum of squares between the observed targets in the dataset,
  * and the targets predicted by the linear approximation.
  *
  * Model always has only one output
  *
  * That is equivalent to `layer.Dense(1, Identity)`
  */
case object LinearRegression extends Model {

  override def build[A: Numeric: Floating: TensorType](x: Expr[A], weights: OutputSeq[A]): Expr[A] = {
    withBias(x, 1f.const.cast[A]) matmul reshape(weights.head).transpose
  }

  override def penalty[E: Numeric : Floating : TensorType](weights: OutputSeq[E]) = zeros[E](Shape())

  private def reshape[A: TensorType](weights: Expr[A]): Expr[A] =
    weights.reshape(1, weights.shape.head)

  override def shapes(features: Int): Seq[Shape] = Seq(Shape(features + 1))

  override def outputs(): Int = 1
}
