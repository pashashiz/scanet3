package scanet.optimizers

import scanet.core._
import scanet.math.syntax._
import scanet.models.Aggregation.Avg
import scanet.models.ParamDef
import scanet.optimizers.SGD.Velocity

case class SGD(rate: Float = 0.01f, momentum: Float = 0.0f, nesterov: Boolean = false)
    extends Algorithm {

  override def params(input: Shape): Params[ParamDef] =
    Params(Velocity -> ParamDef(shape = input, aggregation = Some(Avg)))

  override def build[T: Floating](
      grad: Expr[T],
      params: Params[Expr[T]],
      iter: Expr[Int]): Delta[T] = {
    val vPrev = params(Velocity)
    val m = momentum.const.cast[T]
    val r = rate.const.cast[T]
    val v = (r * grad) - (m * vPrev)
    val delta = if (nesterov) (r * grad) - (m * v) else v
    Delta(delta, Params(Velocity -> v))
  }
}

object SGD {
  val Velocity: Path = "velocity"
}
