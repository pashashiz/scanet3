package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.models.{Activation, Regularization}
import scanet.models.Regularization.Zero
import scanet.syntax._

import scala.collection.immutable.Seq

case class Bias(features: Int, reg: Regularization = Zero) extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.size == 1, "Bias layer can have only one set of weights")
    x + ones[E](features) * weights.head
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = reg.build(weights.head)

  override def outputShape(input: Shape): Shape = input

  override def weightsShapes(input: Shape): Seq[Shape] = Seq(Shape(features))
}
