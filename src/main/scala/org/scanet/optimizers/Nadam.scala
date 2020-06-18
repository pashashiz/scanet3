package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

/** Much like Adam is essentially RMSprop with momentum, Nadam is Adam with Nesterov momentum.
 *
 * @param beta1 The exponential decay rate for the 1st moment estimates.
 * @param beta2 The exponential decay rate for the 2nd moment estimates.
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class Nadam(beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val m = Tensor.zeros[Float](shape).const
    val v = Tensor.zeros[Float](shape).const
    (m zip v).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM, prevV) = meta.unzip
    val m = prevM.decayingAvg(grad, beta1)
    val v = prevV.decayingAvg(grad.sqr, beta2)
    val mNesterov = beta1 * m.boost(beta1, iter) + (1f.const - beta1) * grad.boost(beta1, iter)
    val delta = mNesterov / (v.boost(beta2, iter).sqrt + epsilon)
    Delta(delta, m zip v)
  }
}
object Nadam {

  def apply(beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): Nadam =
    new Nadam(beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)
}