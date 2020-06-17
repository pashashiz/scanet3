package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class Adamax(rate: Output[Float], beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val momentum1 = Tensor.zeros[Float](shape).const
    val momentum2 = Tensor.zeros[Float](shape).const
    (momentum1 zip momentum2).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevMomentum1, prevMomentum2) = meta.unzip
    val momentum1 = prevMomentum1.decayingAvg(grad, beta1)
    val momentum2 = max(beta2 * prevMomentum2, (1f.const - beta2) * grad.abs)
    val delta = rate * momentum1.unbiased(beta1, iter) / (momentum2 + epsilon)
    Delta(delta, momentum1 zip momentum2)
  }
}
object Adamax {

  def apply(rate: Double = 0.001, beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): Adamax =
    new Adamax(rate.toFloat.const, beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)
}