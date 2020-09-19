package org.scanet.models

import org.scanet.core.{Output, OutputSeq, Shape, TensorType}
import org.scanet.math.Floating
import org.scanet.math.Numeric
import org.scanet.math.syntax._

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

  override def build[A: Numeric: Floating: TensorType](x: Output[A], weights: OutputSeq[A]): Output[A] =
    (withBias(x, 1f.const.cast[A]) matmul reshape(weights.head).transpose).sigmoid

  override def penalty[E: Numeric : Floating : TensorType](weights: OutputSeq[E]) = zeros[E](Shape())

  private def reshape[A: TensorType](weights: Output[A]): Output[A] =
    weights.reshape(1, weights.shape.head)

  override def shapes(features: Int): Seq[Shape] = Seq(Shape(features + 1))

  override def outputs(): Int = 1
}
