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
case class AMSGrad(rate: Float = 0.001f, beta1: Float = 0.9f, beta2 : Float = 0.999f, epsilon: Float = 1e-7f) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val m = Tensor.zeros[Float](shape).const
    val vCurrent = Tensor.zeros[Float](shape).const
    val v = Tensor.zeros[Float](shape).const
    zip(m, vCurrent, v).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM, prevV, prevCorrectedVelocity) = meta.unzip3
    val m = prevM.decayingAvg(grad, beta1.const)
    val vCurrent = prevV.decayingAvg(grad.sqr, beta2.const)
    val v = max(prevCorrectedVelocity, vCurrent)
    val delta = rate.const * m / (v.sqrt + epsilon.const)
    Delta(delta, zip(m, vCurrent, v))
  }
}