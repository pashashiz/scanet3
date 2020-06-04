package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class AdaGrad(rate: Float = 1, rho: Float = 0.9F, epsilon: Float = 1e-7F) extends Algorithm {

  def initMeta[X: TensorType](initArg: Tensor[X]): Tensor[Float] = {
    Tensor.zeros[Float](Shape(2 :: initArg.shape.dims))
  }

  def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    // root mean squared
    def rms(x: Output[Float]): Output[Float] =
      (x + epsilon.const).sqrt
    // running (decaying) average
    def avg(prev: Output[Float], curr: Output[Float]): Output[Float] =
      rho.const * prev + (1 - rho).const * curr.pow(2)

    val prevAvgGrad = meta.slice(Projection(0))
    val prevAvgDelta = meta.slice(Projection(1))

    val avgGrad = avg(prevAvgGrad, grad)
    val delta = rate.const * ((rms(prevAvgDelta) / rms(avgGrad)) :* grad)
    val avgDelta = avg(prevAvgDelta, delta)

    // if meta arg is a vector - reshape to matrix to correctly join rows
    val shape = Shape(1 :: meta.shape.dims.tail)
    Delta(delta, avgGrad.reshape(shape).join(avgDelta.reshape(shape)))
  }
}
