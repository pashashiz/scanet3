package org.scanet.optimizers

import org.scanet.core._
import org.scanet.math.{Floating, Numeric}
import org.scanet.syntax._

/**
 * Adam optimization is a stochastic gradient descent method that
 * is based on adaptive estimation of first-order and second-order moments.
 *
 * @param rate learning rate, usually should not be tuned
 * @param beta1 The exponential decay rate for the 1st moment estimates.
 * @param beta2 The exponential decay rate for the 2nd moment estimates.
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class Adam(rate: Float = 0.001f, beta1: Float = 0.9f, beta2 : Float = 0.999f, epsilon: Float = 1e-7f) extends Algorithm  {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    val m = zeros[T](shape)
    val v = zeros[T](shape)
    (m zip v).eval
  }

  override def delta[T: Floating: Numeric: TensorType](grad: Output[T], meta: Output[T], iter: Output[Int]): Delta[T] = {
    val (prevM, prevV) = meta.unzip
    val m = prevM.decayingAvg(grad, beta1.const.cast[T])
    val v = prevV.decayingAvg(grad.sqr, beta2.const.cast[T])
    val delta = (rate.const.cast[T] / (v.boost(beta2.const.cast[T], iter) + epsilon.const.cast[T]).sqrt) * m.boost(beta1.const.cast[T], iter)
    Delta(delta, m zip v)
  }
}