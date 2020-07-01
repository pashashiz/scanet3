package org.scanet.models

import org.scanet.core.{Output, Shape}
import org.scanet.math.syntax._

/** Binary Logistic Regression
 *
 * Similar to Linear Regression but applies a logistic function on top
 * to model a binary dependent variable. Hence, the result is a probability in range `[0, 1]`.
 *
 * Model always has only one output
 */
object LogisticRegression extends Model[Float, Float, Float] {

  override def buildResult(x: Output[Float], weights: Output[Float]): Output[Float] = {
    (withBias(x) * reshape(weights).transpose).sigmoid
  }

  override def buildLoss(x: Output[Float], y: Output[Float], weights: Output[Float]): Output[Float] = {
    val rows = x.shape.dims.head
    // x: (n, m) * wt: (m, 1) -> (n, 1) | y: (n, 1)
    val s = (withBias(x) * reshape(weights).transpose).sigmoid
    val left = y :* s.log
    val right = (1.0f.const - y) :* (1.0f.const - s).log
    (1f / rows).const :* (left.negate - right).sum
  }

  private def reshape(weights: Output[Float]): Output[Float] =
    weights.reshape(1, weights.shape.head)

  override def weightsShape(features: Int): Shape = Shape(features + 1)

  override def outputs(): Int = 1
}
