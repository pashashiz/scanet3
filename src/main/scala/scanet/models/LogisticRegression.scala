package scanet.models

import scanet.core.{Expr, Floating, OutputSeq, Shape, TensorType}
import scanet.math.syntax._

import scala.collection.immutable.Seq

/** Binary Logistic Regression
  *
  * Similar to Linear Regression but applies a logistic function on top
  * to model a binary dependent variable. Hence, the result is a probability in range `[0, 1]`.
  *
  * Model always has only one output
  *
  * That is equivalent to `layer.Dense(1, Sigmoid)`
  */
case object LogisticRegression extends Model {

  override def build[A: Floating](x: Expr[A], weights: OutputSeq[A]): Expr[A] =
    (withBias(x, 1f.const.cast[A]) matmul reshape(weights.head).transpose).sigmoid

  override def penalty[E: Floating](weights: OutputSeq[E]) = zeros[E](Shape())

  private def reshape[A: TensorType](weights: Expr[A]): Expr[A] =
    weights.reshape(1, weights.shape.head)

  override def shapes(features: Int): Seq[Shape] = Seq(Shape(features + 1))

  override def outputs(): Int = 1
}
