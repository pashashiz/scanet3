package org.scanet.optimizers

import org.scanet.core._
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

  def initMeta(shape: Shape): Tensor[Float] = {
    val m = Tensor.zeros[Float](shape).const
    val v = Tensor.zeros[Float](shape).const
    (m zip v).eval
  }

  def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM, prevV) = meta.unzip
    val m = prevM.decayingAvg(grad, beta1.const)
    val v = prevV.decayingAvg(grad.sqr, beta2.const)
    val delta = (rate.const / (v.boost(beta2.const, iter) + epsilon.const).sqrt) :* m.boost(beta1.const, iter)
    Delta(delta, m zip v)
  }
}