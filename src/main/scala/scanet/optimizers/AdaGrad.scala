package scanet.optimizers

import scanet.core._
import scanet.math.{Floating, Numeric}
import scanet.syntax._

/** Adagrad is an optimizer with parameter-specific learning rates,
  * which are adapted relative to how frequently a parameter gets updated during training.
  * The more updates a parameter receives, the smaller the updates.
  *
  * @param rate the learning rate.
  * @param epsilon a constant epsilon used to better conditioning the grad update
  */
case class AdaGrad(rate: Float = 0.001f, epsilon: Float = 1e-7f) extends Algorithm {

  override def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T] = {
    Tensor.zeros[T](shape)
  }

  override def delta[T: Floating: Numeric: TensorType](
      grad: Expr[T],
      prevGradAcc: Expr[T],
      iter: Expr[Int]): Delta[T] = {
    // we accumulate all squared gradient per each weight
    val gradAcc = prevGradAcc + grad.sqr
    // the larger gradient is accumulated the lower rate is applied for a given weight
    val rates = rate.const.cast[T] / gradAcc.sqrtZeroSafe(epsilon.const.cast[T])
    val delta = rates * grad
    Delta(delta, gradAcc)
  }
}
