package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.models.Activation

/** A layer which applies activation function to the input.
  *
  * Could also be constructed as: {{{activation.layer}}}
  *
  * @param activation activation function
  */
case class Activate(activation: Activation) extends StatelessLayer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.isEmpty, "Activate layer does not require weights")
    activation.build(x)
  }

  override def outputShape(input: Shape): Shape = input

  override def toString: String = activation.toString
}
