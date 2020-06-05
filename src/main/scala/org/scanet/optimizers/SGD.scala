package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor}
import org.scanet.math.syntax._

case class SGD(rate: Double = 0.01, momentum: Double = 0.0, nesterov: Boolean = false) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros(shape)

  override def delta(grad: Output[Float], meta: Output[Float]): Delta = {
    val m = momentum.toFloat.const
    val r = rate.toFloat.const
    val v = r * grad - m * meta
    val d = if (nesterov) r * grad - m * v else v
    Delta(d, v)
  }
}
