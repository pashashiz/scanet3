package scanet.optimizers

import scanet.core._
import scanet.models.Aggregation.Avg
import scanet.models.{Initializer, ParamDef}
import scanet.optimizers.Adamax.{Momentum1, Momentum2}
import scanet.syntax._

/** Adamax is a variant of Adam based on the infinity norm.
  * It is sometimes superior to adam, specially in models with embeddings.
  *
  * In contrast to Adam, the sparse implementation of this algorithm only updates variable slices
  * and corresponding `m` and `v` terms when that part of the variable was used in the forward pass.
  * This means that the sparse behavior is contrast to the dense behavior (similar to some momentum
  * implementations which ignore momentum unless a variable slice was actually used).
  *
  * @param rate learning rate
  * @param beta1 The exponential decay rate for the 1st moment estimates
  * @param beta2 The exponential decay rate for the exponentially weighted infinity norm
  * @param epsilon a constant epsilon used to better conditioning the grad update
  */
case class Adamax(
    rate: Float = 0.001f,
    beta1: Float = 0.9f,
    beta2: Float = 0.999f,
    epsilon: Float = 1e-7f,
    initAcc: Float = 0.001f)
    extends Algorithm {

  override def params(input: Shape): Params[ParamDef] =
    Params(
      Momentum1 -> ParamDef(
        shape = input,
        initializer = Initializer.Const(initAcc),
        aggregation = Some(Avg)),
      Momentum2 -> ParamDef(
        shape = input,
        initializer = Initializer.Const(initAcc),
        aggregation = Some(Avg)))

  override def build[T: Floating](
      grad: Expr[T],
      params: Params[Expr[T]],
      iter: Expr[Int]): Delta[T] = {
    val prevM1 = params(Momentum1)
    val prevM2 = params(Momentum2)
    val m1 = prevM1.decayingAvg(grad, beta1.const.cast[T])
    val m2 = max(beta2.const.cast[T] * prevM2, (1f.const.cast[T] - beta2.const.cast[T]) * grad.abs)
    val delta =
      rate.const.cast[T] * m1.boost(beta1.const.cast[T], iter) / (m2 + epsilon.const.cast[T])
    Delta(delta, Params(Momentum1 -> m1, Momentum2 -> m2))
  }
}

object Adamax {
  val Momentum1: Path = "momentum_1"
  val Momentum2: Path = "momentum_2"
}
