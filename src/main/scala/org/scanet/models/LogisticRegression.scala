package org.scanet.models

import org.scanet.core.{Output, Shape, TensorType}
import org.scanet.math.Floating
import org.scanet.math.Numeric
import org.scanet.math.syntax._

/** Binary Logistic Regression
 *
 * Similar to Linear Regression but applies a logistic function on top
 * to model a binary dependent variable. Hence, the result is a probability in range `[0, 1]`.
 *
 * Model always has only one output
 */
case class LogisticRegression[E: Floating: Numeric: TensorType]() extends Model[E, E] {

  override def buildResult(x: Output[E], weights: Output[E]): Output[E] = {
    (withBias(x) * reshape(weights).transpose).sigmoid
  }

  override def buildLoss(x: Output[E], y: Output[E], weights: Output[E]): Output[E] = {
    val rows = x.shape.dims.head
    // x: (n, m) * wt: (m, 1) -> (n, 1) | y: (n, 1)
    val one = 1.0f.const.cast[E]
    val s = (withBias(x) * reshape(weights).transpose).sigmoid
    val left = y :* s.log
    val right = (one - y) :* (one - s).log
    (1f / rows).const.cast[E] :* (left.negate - right).sum
  }

  private def reshape(weights: Output[E]): Output[E] =
    weights.reshape(1, weights.shape.head)

  override def weightsShape(features: Int): Shape = Shape(features + 1)

  override def outputs(): Int = 1
}
