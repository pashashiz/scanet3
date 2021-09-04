package org.scanet.optimizers

import org.scanet.core.{Expr, Shape, Tensor, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

case class SGD(rate: Float = 0.01f, momentum: Float = 0.0f, nesterov: Boolean = false)
    extends Algorithm {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    Tensor.zeros(shape)
  }

  override def delta[T: Floating: Numeric: TensorType](
      grad: Expr[T],
      meta: Expr[T],
      iter: Expr[Int]): Delta[T] = {
    val m = momentum.const.cast[T]
    val r = rate.const.cast[T]
    val v = (r * grad) - (m * meta)
    val delta = if (nesterov) (r * grad) - (m * v) else v
    Delta(delta, v)
  }
}
