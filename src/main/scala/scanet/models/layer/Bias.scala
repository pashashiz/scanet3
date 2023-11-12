package scanet.models.layer

import scanet.core.Params._
import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.syntax.zeros
import scanet.models.Aggregation.Avg
import scanet.models.Initializer.Zeros
import scanet.models.Regularization.Zero
import scanet.models.{Initializer, Model, ParamDef, Regularization}
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
case class Bias(
    features: Int,
    reg: Regularization = Zero,
    initializer: Initializer = Zeros,
    override val trainable: Boolean = true)
    extends StatelessLayer {

  override def params(input: Shape): Params[ParamDef] =
    Params(Weights -> ParamDef(Shape(features), initializer, Some(Avg), trainable = trainable))

  override def buildStateless[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] =
    input + params.weights

  override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] =
    if (trainable) reg.build(params.weights) else zeros[E](Shape())

  override def outputShape(input: Shape): Shape = input

  override def makeTrainable(trainable: Boolean): Bias = copy(trainable = trainable)

  override def toString: String = s"Bias($features)"
}
