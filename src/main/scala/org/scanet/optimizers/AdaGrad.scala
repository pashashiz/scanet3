package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class AdaGrad(rate: Float = 1, rho: Float = 0.9f) extends Algorithm {

  def initMeta[X: TensorType](initArg: Tensor[X]): Tensor[Float] = {
    Tensor.zeros[Float](Shape(2 :: initArg.shape.dims))
  }

  def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    // todo
    ???
  }
}
