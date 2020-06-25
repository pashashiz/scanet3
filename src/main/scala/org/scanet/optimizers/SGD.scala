package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor}
import org.scanet.math.syntax._

case class SGD(rate: Float = 0.01f, momentum: Float = 0.0f, nesterov: Boolean = false) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros(shape)

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val m = momentum.const
    val r = rate.const
    val v = r * grad - m * meta
    val d = if (nesterov) r * grad - m * v else v
    Delta(d, v)
  }
}
