package org.scanet.optimizers

import org.scanet.core._
import org.scanet.math.{Floating, Numeric}
import org.scanet.syntax._

/** Much like Adam is essentially RMSprop with momentum, Nadam is Adam with Nesterov momentum.
 *
 * NOTE: we might need to reimplement it, see
 *  - https://pywick.readthedocs.io/en/stable/_modules/pywick/optimizers/nadam.html
 *  - https://github.com/uralik/keras/blob/c89b7f632a2883d0742d9d89ab959e54f7ce0dd0/keras/optimizers.py
 *
 * @param beta1 The exponential decay rate for the 1st moment estimates.
 * @param beta2 The exponential decay rate for the 2nd moment estimates.
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class Nadam(beta1: Float = 0.9f, beta2 : Float = 0.999f, epsilon: Float = 1e-7f, initAcc: Float = 0.001f) extends Algorithm {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    val m = fill[Float](shape)(initAcc).cast[T]
    val v = fill[Float](shape)(initAcc).cast[T]
    (m zip v).eval
  }

  override def delta[T: Floating: Numeric: TensorType](grad: Output[T], meta: Output[T], iter: Output[Int]): Delta[T] = {
    val (prevM, prevV) = meta.unzip
    val m = prevM.decayingAvg(grad, beta1.const.cast[T])
    val v = prevV.decayingAvg(grad.sqr, beta2.const.cast[T])
    val mNesterov = (beta1.const.cast[T] * m.boost(beta1.const.cast[T], iter)) +
      ((1f.const.cast[T] - beta1.const.cast[T]) * grad.boost(beta1.const.cast[T], iter))
    val delta = mNesterov / (v.boost(beta2.const.cast[T], iter).sqrt + epsilon.const.cast[T])
    Delta(delta, m zip v)
  }
}