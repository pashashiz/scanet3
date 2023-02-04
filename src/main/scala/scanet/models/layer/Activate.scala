package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.models.Activation
import scala.collection.immutable.Seq

/** A layer which applies activation function to the input.
  *
  * Could also be constructed as: {{{activation.layer}}}
  *
  * @param activation activation function
  */
case class Activate(activation: Activation) extends WeightlessLayer {

  override def name: String = activation.toString

  override def build[E: Floating](input: Expr[E], weights: Seq[Expr[E]]): Expr[E] = {
    require(weights.isEmpty, "Activate layer does not require weights")
    activation.build(input)
  }

  override def outputShape(input: Shape): Shape = input

  override def toString: String = activation.toString
}
