package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.models.Initializer.Zeros
import scanet.models.{Initializer, Regularization}
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
  * @param initializer kernel initializer
  */
case class Bias(features: Int, reg: Regularization = Zero, initializer: Initializer = Zeros)
    extends StatelessLayer {

  override def build[E: Floating](input: Expr[E], weights: Seq[Expr[E]]): Expr[E] = {
    require(weights.size == 1, "Bias layer can have only one set of weights")
    input + weights.head
  }

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E] = reg.build(weights.head)

  override def weightsShapes(input: Shape): Seq[Shape] = Seq(Shape(features))

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] =
    Seq(initializer.build[E](weightsShapes(input).head))

  override def outputShape(input: Shape): Shape = input
}
