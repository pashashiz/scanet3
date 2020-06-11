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
case class Adam(rate: Output[Float], beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm  {

  def initMeta(shape: Shape): Tensor[Float] = {
    val momentum = Tensor.zeros[Float](shape).const
    val velocity = Tensor.zeros[Float](shape).const
    (momentum zip velocity).eval
  }

  def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    val iter = 3.0f // todo: need iter here
    val (prevMomentum, prevVelocity) = meta.unzip
    val momentum = prevMomentum.decayingAvg(grad, beta1)
    val momentumUnbiased = momentum / (1.0f.const - beta1.pow(iter))
    val velocity = prevVelocity.decayingAvg(grad.sqr, beta2)
    val velocityUnbiased = velocity / (1.0f.const - beta2.pow(iter))
    val delta = (rate / (velocityUnbiased + epsilon).sqrt) :* momentumUnbiased
    Delta(delta, momentum zip velocity)
  }
}

object Adam {

  def apply(rate: Double = 0.001, beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): Adam =
    new Adam(rate.toFloat.const, beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)

}
