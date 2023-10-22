package scanet.optimizers

import scanet.core._
import scanet.models.Aggregation.Avg
import scanet.models.ParamDef
import scanet.optimizers.AdaGrad.GradAcc
import scanet.syntax._

/** Adagrad is an optimizer with parameter-specific learning rates,
  * which are adapted relative to how frequently a parameter gets updated during training.
  * The more updates a parameter receives, the smaller the updates.
  *
  * @param rate the learning rate.
  * @param epsilon a constant epsilon used to better conditioning the grad update
  */
case class AdaGrad(rate: Float = 0.001f, epsilon: Float = 1e-7f) extends Algorithm {

  override def params(input: Shape): Params[ParamDef] =
    Params(GradAcc -> ParamDef(shape = input, aggregation = Some(Avg)))

  override def build[T: Floating](
      grad: Expr[T],
      params: Params[Expr[T]],
      iter: Expr[Int]): Delta[T] = {
    val prevGradAcc = params(GradAcc)
    // we accumulate all squared gradient per each weight
    val gradAcc = prevGradAcc + grad.sqr
    // the larger gradient is accumulated the lower rate is applied for a given weight
    val rates = rate.const.cast[T] / gradAcc.sqrtZeroSafe(epsilon.const.cast[T])
    val delta = rates * grad
    Delta(delta, Params(GradAcc -> gradAcc))
  }
}

object AdaGrad {
  val GradAcc: Path = "grad_acc"
}
