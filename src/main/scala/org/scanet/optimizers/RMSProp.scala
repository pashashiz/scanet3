package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

/**
 * RMSprop, similar to Adadelta, is an improvement upon Adagrad's to resolve radically diminishing learning rates.
 * It is implemented by:
 * - maintain a moving (discounted) average of the square of gradients
 * - divide gradient by the root of this average
 *
 * @param rate learning rate
 * @param rho the decay rate
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class RMSProp(rate: Float = 0.001f, rho: Float = 0.9f, epsilon: Float = 1e-7f) extends Algorithm {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    Tensor.zeros[T](shape)
  }

  override def delta[T: Floating: Numeric: TensorType](grad: Output[T], prevAvgGrad: Output[T], iter: Output[Int]): Delta[T] = {
    val avgGrad = prevAvgGrad.decayingAvg(grad.sqr, rho.const.cast[T])
    val delta = (rate.const.cast[T] / avgGrad.sqrtZeroSafe(epsilon.const.cast[T])) :* grad
    Delta(delta, avgGrad)
  }
}