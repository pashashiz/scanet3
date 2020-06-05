package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class AdaDelta(rate: Float = 1, rho: Float = 0.9f) extends Algorithm {

  def initMeta(shape: Shape): Tensor[Float] = {
    val arg = Tensor.zeros[Float](shape).const
    arg.zip(arg).eval
  }

  def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    // root mean squared
    def rms(x: Output[Float]): Output[Float] =
      (x + 1e-7f.const).sqrt
    // running (decaying) average
    def avg(prev: Output[Float], curr: Output[Float]): Output[Float] =
      rho.const * prev + (1 - rho).const * curr.sqr

    val (prevAvgGrad, prevAvgDelta) = meta.unzip
    val avgGrad = avg(prevAvgGrad, grad)
    val delta = rate.const * ((rms(prevAvgDelta) / rms(avgGrad)) :* grad)
    val avgDelta = avg(prevAvgDelta, delta)
    Delta(delta, avgGrad.zip(avgDelta))
  }
}
