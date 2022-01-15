package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.models.Regularization
import scanet.models.Regularization.Zero
import scanet.syntax._

import scala.collection.immutable.Seq

/** A layer which sums up a bias vector (weights) with the input.
  * When input has rank > 1 the summation will be broadcasted.
  *
  * Given an input `Shape(N, ..., features)` and a bias `Shape(features)`
  * each bias `[i-th]` element will be added to every `[N, ..., i-th]` element
  *
  * @param features the number of features
  * @param reg regularization
  */
case class Bias(features: Int, reg: Regularization = Zero) extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.size == 1, "Bias layer can have only one set of weights")
    x + weights.head
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = reg.build(weights.head)

  override def outputShape(input: Shape): Shape = input

  override def weightsShapes(input: Shape): Seq[Shape] = Seq(Shape(features))
}
