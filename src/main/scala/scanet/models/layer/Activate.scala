package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.syntax.zeros
import scanet.models.Activation

import scala.collection.immutable.Seq

case class Activate(activation: Activation) extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.isEmpty, "Activate layer does not require weights")
    activation.build(x)
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())

  override def outputShape(input: Shape): Shape = input

  override def weightsCount: Int = 0

  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty

  override def toString: String = activation.toString
}
