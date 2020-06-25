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
case class RMSProp(rate: Float = 0.001f, rho: Float = 0.9f, epsilon: Float = 1e-7f) extends Algorithm {

  def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros(shape)

  def delta(grad: Output[Float], prevAvgGrad: Output[Float], iter: Output[Int]): Delta = {
    val avgGrad = prevAvgGrad.decayingAvg(grad.sqr, rho.const)
    val delta = (rate.const / avgGrad.rms(epsilon.const)) :* grad
    Delta(delta, avgGrad)
  }
}