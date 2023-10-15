package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.models.Activation

/** A layer which applies activation function to the input.
  *
  * Could also be constructed as: {{{activation.layer}}}
  *
  * @param activation activation function
  */
case class Activate(activation: Activation) extends NotTrainableLayer {

  override def name: String = activation.toString

  override def build_[E: Floating](input: Expr[E]): Expr[E] =
    activation.build(input)

  override def outputShape(input: Shape): Shape = input

  override def toString: String = activation.toString
}
