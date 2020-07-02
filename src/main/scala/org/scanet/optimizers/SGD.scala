package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

case class SGD(rate: Float = 0.01f, momentum: Float = 0.0f, nesterov: Boolean = false) extends Algorithm {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    Tensor.zeros(shape)
  }

  override def delta[T: Floating: Numeric: TensorType](grad: Output[T], meta: Output[T], iter: Output[Int]): Delta[T] = {
    val m = momentum.const.cast[T]
    val r = rate.const.cast[T]
    val v = r * grad - m * meta
    val d = if (nesterov) r * grad - m * v else v
    Delta(d, v)
  }
}
