package scanet.optimizers

import scanet.core._
import scanet.models.Aggregation.Avg
import scanet.models.ParamDef
import scanet.optimizers.Adam.{Momentum, Velocity}
import scanet.syntax._

/** Adam optimization is a stochastic gradient descent method that
  * is based on adaptive estimation of first-order and second-order moments.
  *
  * The update follows [Kingma and Ba, 2014](https://arxiv.org/abs/1412.6980):
  * {{{
  *   m = beta1 * m + (1 - beta1) * g
  *   v = beta2 * v + (1 - beta2) * g^2
  *   delta = rate * m^ / (sqrt(v^) + epsilon)
  * }}}
  * where `m^ = m / (1 - beta1^t)` and `v^ = v / (1 - beta2^t)` are bias corrected moments.
  *
  * @param rate learning rate, usually should not be tuned
  * @param beta1 The exponential decay rate for the 1st moment estimates.
  * @param beta2 The exponential decay rate for the 2nd moment estimates.
  * @param epsilon a constant epsilon used to better conditioning the grad update
  */
case class Adam(
    rate: Float = 0.001f,
    beta1: Float = 0.9f,
    beta2: Float = 0.999f,
    epsilon: Float = 1e-8f)
    extends Algorithm {

  override def params(input: Shape): Params[ParamDef] =
    Params(
      Momentum -> ParamDef(shape = input, aggregation = Some(Avg)),
      Velocity -> ParamDef(shape = input, aggregation = Some(Avg)))

  override def build[T: Floating](
      grad: Expr[T],
      params: Params[Expr[T]],
      iter: Expr[Int]): Delta[T] = {
    val prevM = params(Momentum)
    val prevV = params(Velocity)
    val m = prevM.decayingAvg(grad, beta1.const.cast[T])
    val v = prevV.decayingAvg(grad.sqr, beta2.const.cast[T])
    val mHat = m.boost(beta1.const.cast[T], iter)
    val vHat = v.boost(beta2.const.cast[T], iter)
    val delta = rate.const.cast[T] * mHat / (vHat.sqrt + epsilon.const.cast[T])
    Delta(delta, Params(Momentum -> m, Velocity -> v))
  }
}

object Adam {
  val Momentum: Path = "momentum"
  val Velocity: Path = "velocity"
}
