package scanet.models

import scanet.core.{Expr, Floating, OutputSeq, Shape, TensorType}
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

  override def build[A: Floating](x: Expr[A], weights: OutputSeq[A]): Expr[A] = {
    withBias(x, 1f.const.cast[A]) matmul reshape(weights.head).transpose
  }

  override def penalty[E: Floating](weights: OutputSeq[E]) = zeros[E](Shape())

  private def reshape[A: TensorType](weights: Expr[A]): Expr[A] =
    weights.reshape(1, weights.shape.head)

  override def weightsShapes(input: Shape): Seq[Shape] = {
    require(input.rank == 1, "features should have a shape (features)")
    Seq(Shape(input(0) + 1))
  }

  override def outputShape(input: Shape): Shape = Shape(1)
}
