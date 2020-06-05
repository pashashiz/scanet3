package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

/**
 * `Adadelta` optimization is a stochastic gradient descent method that is based on adaptive
 * learning rate per dimension to address two drawbacks:
 * - The continual decay of learning rates throughout training
 * - The need for a manually selected global learning rate
 *
 * `Adadelta` is a more robust extension of `Adagrad` that adapts learning rates based
 * on a moving window of gradient updates, instead of accumulating all past gradients. T
 * his way, `Adadelta` continues learning even when many updates have been done.
 * Compared to `Adagrad`, in the original version of `Adadelta` you don't have to set an initial learning rate.
 *
 * In this version, initial learning rate can be set, as in most other optimizers.
 *
 * @param rate learning rate, usually should not be tuned
 * @param rho the decay rate
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class AdaDelta(rate: Double = 1.0, rho: Double = 0.9, epsilon: Double = 1e-7) extends Algorithm {

  def initMeta(shape: Shape): Tensor[Float] = {
    val arg1 = Tensor.zeros[Float](shape).const
    val arg2 = Tensor.zeros[Float](shape).const
    arg1.zip(arg2).eval
  }

  def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    val (prevAvgGrad, prevAvgDelta) = meta.unzip
    val avgGrad = avg(prevAvgGrad, grad)
    val delta = rate.toFloat.const * ((rms(prevAvgDelta) / rms(avgGrad)) :* grad)
    val avgDelta = avg(prevAvgDelta, delta)
    Delta(delta, avgGrad.zip(avgDelta))
  }
}
