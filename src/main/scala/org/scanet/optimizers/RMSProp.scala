package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor}
import org.scanet.math.syntax._

case class RMSProp(rate: Float = 0.001f, rho: Float = 0.9f) extends Algorithm with RMS {

  def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros(shape)

  def delta(grad: Output[Float], prevAvgGrad: Output[Float]): Delta = {
    val avgGrad = avg(prevAvgGrad, grad)
    val delta = (rate.const / rms(avgGrad)) :* grad
    Delta(delta, avgGrad)
  }
}
