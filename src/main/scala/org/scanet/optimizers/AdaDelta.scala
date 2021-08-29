package org.scanet.optimizers

import org.scanet.core._
import org.scanet.math.{Floating, Numeric}
import org.scanet.syntax._

/** `Adadelta` optimization is a stochastic gradient descent method that is based on adaptive
  * learning rate per dimension to address two drawbacks:
  * - The continual decay of learning rates throughout training
  * - The need for a manually selected global learning rate
  *
  * `Adadelta` is a more robust extension of `Adagrad` that adapts learning rates based
  * on a moving window of gradient updates, instead of accumulating all past gradients.
  * This way, `Adadelta` continues learning even when many updates have been done.
  * Compared to `Adagrad`, in the original version of `Adadelta` you don't have to set an initial learning rate.
  *
  * In this version, initial learning rate can be set, as in most other optimizers.
  *
  * @param rate learning rate, usually should not be tuned
  * @param rho the decay rate
  * @param epsilon a constant epsilon used to better conditioning the grad update
  */
case class AdaDelta(rate: Float = 1.0f, rho: Float = 0.9f, epsilon: Float = 1e-7f)
    extends Algorithm {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    val avgGrad = zeros[T](shape)
    val avgDelta = zeros[T](shape)
    (avgGrad zip avgDelta).eval
  }

  override def delta[T: Floating: Numeric: TensorType](
      grad: Expr[T],
      meta: Expr[T],
      iter: Expr[Int]): Delta[T] = {
    val (prevAvgGrad, prevAvgDelta) = meta.unzip
    val avgGrad = prevAvgGrad.decayingAvg(grad.sqr, rho.const.cast[T])
    val delta = rate.const.cast[T] * ((prevAvgDelta.sqrtZeroSafe(epsilon.const.cast[T]) / avgGrad
      .sqrtZeroSafe(epsilon.const.cast[T])) * grad)
    val avgDelta = prevAvgDelta.decayingAvg(delta.sqr, rho.const.cast[T])
    Delta(delta, avgGrad zip avgDelta)
  }
}
