package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

/** AMSGrad optimizer which is based on Adam and RMSProp.
 *
 * The key difference of AMSGrad with Adam is that it maintains the maximum of all `v`
 * until the present time step and uses this maximum value for normalizing the running average
 * of the gradient instead of `v` in Adam. By doing this, AMSGrad results in a non-increasing
 * step size and avoids the pitfalls of Adam and RMSProp.
 *
 * @param rate learning rate, usually should not be tuned
 * @param beta1 The exponential decay rate for the 1st moment estimates.
 * @param beta2 The exponential decay rate for the 2nd moment estimates.
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class AMSGrad(rate: Output[Float], beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val m = Tensor.zeros[Float](shape).const
    val vCurrent = Tensor.zeros[Float](shape).const
    val v = Tensor.zeros[Float](shape).const
    zip(m, vCurrent, v).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM, prevV, prevCorrectedVelocity) = meta.unzip3
    val m = prevM.decayingAvg(grad, beta1)
    val vCurrent = prevV.decayingAvg(grad.sqr, beta2)
    val v = max(prevCorrectedVelocity, vCurrent)
    val delta = rate * m / (v.sqrt + epsilon)
    Delta(delta, zip(m, vCurrent, v))
  }
}
object AMSGrad {

  def apply(rate: Double = 0.001, beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): AMSGrad =
    new AMSGrad(rate.toFloat.const, beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)
}