package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class AMSGrad(rate: Output[Float], beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val momentum = Tensor.zeros[Float](shape).const
    val velocity = Tensor.zeros[Float](shape).const
    val correctedVelocity = Tensor.zeros[Float](shape).const
    val suffix = Tensor.zeros[Float](shape).const
    ((momentum zip velocity) zip (correctedVelocity zip suffix)).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (args1, args2) = meta.unzip
    val (prevMomentum, prevVelocity) = args1.unzip
    val (prevCorrectedVelocity, suffix) = args2.unzip
    val momentum = prevMomentum.decayingAvg(grad, beta1)
    val velocity = prevVelocity.decayingAvg(grad.sqr, beta2)
    val correctedVelocity = max(prevCorrectedVelocity, velocity)
    val delta = rate * momentum / (correctedVelocity.sqrt + epsilon)
    Delta(delta, (momentum zip velocity) zip (correctedVelocity zip suffix))
  }
}
object AMSGrad {

  def apply(rate: Double = 0.001, beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): AMSGrad =
    new AMSGrad(rate.toFloat.const, beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)
}