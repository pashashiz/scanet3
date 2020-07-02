package org.scanet.optimizers

import org.scanet.core._
import org.scanet.math.{Floating, Numeric}
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

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    val m = Tensor.zeros[T](shape).const
    val vCurrent = Tensor.zeros[T](shape).const
    val v = Tensor.zeros[T](shape).const
    zip(m, vCurrent, v).eval
  }

  override def delta[T: Floating: Numeric: TensorType](grad: Output[T], meta: Output[T], iter: Output[Int]): Delta[T] = {
    val (prevM, prevV, prevCorrectedVelocity) = meta.unzip3
    val m = prevM.decayingAvg(grad, beta1.const.cast[T])
    val vCurrent = prevV.decayingAvg(grad.sqr, beta2.const.cast[T])
    val v = max(prevCorrectedVelocity, vCurrent)
    val delta = rate.const.cast[T] * m / (v.sqrt + epsilon.const.cast[T])
    Delta(delta, zip(m, vCurrent, v))
  }
}