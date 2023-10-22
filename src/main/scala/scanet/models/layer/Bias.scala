package scanet.models.layer

import scanet.core.Params._
import scanet.core.{Expr, Floating, Params, Shape}
import scanet.models.Aggregation.Avg
import scanet.models.Initializer.Zeros
import scanet.models.Regularization.Zero
import scanet.models.{Initializer, ParamDef, Regularization}
import scanet.syntax._

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

  override def params(input: Shape): Params[ParamDef] =
    Params(Weights -> ParamDef(Shape(features), initializer, Some(Avg), trainable = true))

  override def buildStateless_[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] =
    input + params.weights

  override def penalty[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] =
    reg.build(params.weights)

  override def outputShape(input: Shape): Shape = input

}
