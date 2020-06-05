package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class AdaDelta(rate: Float = 1, rho: Float = 0.9f) extends Algorithm with RMS {

  def initMeta(shape: Shape): Tensor[Float] = {
    val arg = Tensor.zeros[Float](shape).const
    arg.zip(arg).eval
  }

  def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    val (prevAvgGrad, prevAvgDelta) = meta.unzip
    val avgGrad = avg(prevAvgGrad, grad)
    val delta = rate.const * ((rms(prevAvgDelta) / rms(avgGrad)) :* grad)
    val avgDelta = avg(prevAvgDelta, delta)
    Delta(delta, avgGrad.zip(avgDelta))
  }
}
