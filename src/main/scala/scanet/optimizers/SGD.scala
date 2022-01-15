package scanet.optimizers

import scanet.core.{Expr, Floating, Shape, Tensor}
import scanet.math.syntax._

case class SGD(rate: Float = 0.01f, momentum: Float = 0.0f, nesterov: Boolean = false)
    extends Algorithm {

  override def initMeta[T: Floating](shape: Shape): Tensor[T] = {
    Tensor.zeros(shape)
  }

  override def delta[T: Floating](
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
