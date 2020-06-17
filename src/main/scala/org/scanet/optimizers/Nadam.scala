package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class Nadam(rate: Output[Float], beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val momentum = Tensor.zeros[Float](shape).const
    val velocity = Tensor.zeros[Float](shape).const
    (momentum zip velocity).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevMomentum, prevVelocity) = meta.unzip
    val momentum = prevMomentum.decayingAvg(grad, beta1)
    val velocity = prevVelocity.decayingAvg(grad.sqr, beta2)
    val momentumNesterov = beta1 * momentum.unbiased(beta1, iter) + (1f.const - beta1) * grad.unbiased(beta1, iter)
    val delta = momentumNesterov / (velocity.unbiased(beta2, iter).sqrt + epsilon)
    Delta(delta, momentum zip velocity)
  }
}
object Nadam {

  def apply(rate: Double = 0.001, beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): Nadam =
    new Nadam(rate.toFloat.const, beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)
}