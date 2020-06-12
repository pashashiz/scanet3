package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor}
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
case class RMSProp(rate: Output[Float], rho: Output[Float], epsilon: Output[Float]) extends Algorithm {

  def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros(shape)

  def delta(grad: Output[Float], prevAvgGrad: Output[Float], iter: Output[Int]): Delta = {
    val avgGrad = prevAvgGrad.decayingAvg(grad.sqr, rho)
    val delta = (rate / avgGrad.rms(epsilon)) :* grad
    Delta(delta, avgGrad)
  }
}
object RMSProp {

  def apply(rate: Double = 0.001, rho: Double = 0.9, epsilon: Double = 1e-7): RMSProp = {
    RMSProp(rate.toFloat.const, rho.toFloat.const, epsilon.toFloat.const)
  }
}
