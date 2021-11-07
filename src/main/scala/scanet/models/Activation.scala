package scanet.models

import scanet.core.{Expr, TensorType}
import scanet.math.{Floating, Numeric}
import scanet.math.syntax._
import scala.collection.immutable.Seq

/** NOTE: activation might be expressed as a function Output[A] => Output[A]
  * however we cannot express that easily in scala 2 cause of eta expansion,
  * basically functions in scala are monomorphic and if we go that what
  * we would have to explicitly specify type each time we initialize it
  * or have a context to perform type inference and something like this would be impossible:
  * {{{
  * Dense(4, Sigmoid) >> Dense(1, Sigmoid)
  * }}}
  */
trait Activation {
  def build[A: Numeric: Floating: TensorType](in: Expr[A]): Expr[A]
}

object Activation {

  /** Identity activation function, the output is equal to input.
    *
    * Identity is usually used for linear regression.
    */
  case object Identity extends Activation {
    override def build[A: Numeric: Floating: TensorType](in: Expr[A]): Expr[A] = in
  }

  /** Sigmoid activation function, {{{sigmoid(x) = 1 / (1 + exp(-x))}}}.
    *
    * Applies the sigmoid activation function.
    * For small values `< -5`, sigmoid returns a value close to `0`,
    * and for large values `> 5` the result of the function gets close to `1`.
    *
    * Sigmoid is equivalent to a 2-element `Softmax`, where the second element is assumed to be zero.
    * The sigmoid function always returns a value between `0` and `1`.
    */
  case object Sigmoid extends Activation {
    override def build[A: Numeric: Floating: TensorType](in: Expr[A]): Expr[A] = in.sigmoid
  }

  /** Hyperbolic tangent activation function
    */
  case object Tanh extends Activation {
    override def build[A: Numeric: Floating: TensorType](in: Expr[A]): Expr[A] = in.tanh
  }

  /** Softmax converts a real vector to a vector of categorical probabilities.
    *
    * The elements of the output vector are in range `(0, 1)` and sum to `1`.
    *
    * Softmax is often used as the activation for the last layer of a classification network
    * because the result could be interpreted as a probability distribution.
    */
  case object Softmax extends Activation {
    override def build[A: Numeric: Floating: TensorType](in: Expr[A]): Expr[A] = {
      val e = in.exp
      val sum = e.sum(axis = Seq(1))
      e / sum.reshape(sum.shape :+ 1)
    }
  }

  /** Applies the rectified linear unit activation function.
    *
    * With default values, this returns the standard ReLU activation: {{{max(x, 0)}}}
    * the element-wise maximum of `0` and the `input tensor`.
    *
    * Modifying default parameters allows you to use non-zero thresholds
    *
    * @param threshold max threshold
    */
  case class ReLU(threshold: Float = 0f) extends Activation {
    override def build[A: Numeric: Floating: TensorType](in: Expr[A]) =
      max(threshold.const.cast[A], in)
  }
}
