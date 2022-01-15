package scanet.models

import scanet.core.{Expr, Floating, Shape}
import scanet.math.syntax._

/** Function to compute regularization for a given weight tensor.
  * Regularization value should be a scalar.
  *
  * Regularization allow you to apply penalties on layer parameters.
  * These penalties are summed into the loss function that the network optimizes.
  */
trait Regularization extends Serializable {
  def build[A: Floating](weights: Expr[A]): Expr[A]
}

object Regularization {

  /** Absence of Regularization
    */
  case object Zero extends Regularization {
    override def build[A: Floating](weights: Expr[A]) = zeros[A](Shape())
  }

  /** A regularizer that applies a `L1` regularization penalty.
    *
    * The `L1` regularization penalty is computed as {{{lambda * weights.abs.sum / 2}}}
    *
    * @param lambda regularization factor
    */
  case class L1(lambda: Float = 0.01f) extends Regularization {
    override def build[A: Floating](weights: Expr[A]) =
      lambda.const.cast[A] * weights.abs.sum / 2f.const.cast[A]
  }

  /** A regularizer that applies a `L2` regularization penalty.
    *
    * The `L2` regularization penalty is computed as {{{lambda * weights.sqr.sum / 2}}}
    *
    * @param lambda regularization factor
    */
  case class L2(lambda: Float = 0.01f) extends Regularization {
    override def build[A: Floating](weights: Expr[A]) =
      lambda.const.cast[A] * weights.sqr.sum / 2f.const.cast[A]
  }
}
