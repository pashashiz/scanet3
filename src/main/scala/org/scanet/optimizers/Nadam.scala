package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

/** Much like Adam is essentially RMSprop with momentum, Nadam is Adam with Nesterov momentum.
 *
 * @param beta1 The exponential decay rate for the 1st moment estimates.
 * @param beta2 The exponential decay rate for the 2nd moment estimates.
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class Nadam(beta1: Float = 0.9f, beta2 : Float = 0.999f, epsilon: Float = 1e-7f) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val m = Tensor.zeros[Float](shape).const
    val v = Tensor.zeros[Float](shape).const
    (m zip v).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM, prevV) = meta.unzip
    val m = prevM.decayingAvg(grad, beta1.const)
    val v = prevV.decayingAvg(grad.sqr, beta2.const)
    val mNesterov = beta1.const * m.boost(beta1.const, iter) + (1f.const - beta1.const) * grad.boost(beta1.const, iter)
    val delta = mNesterov / (v.boost(beta2.const, iter).sqrt + epsilon.const)
    Delta(delta, m zip v)
  }
}