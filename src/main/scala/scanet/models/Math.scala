package scanet.models

import scanet.core.Params.Weights
import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.syntax._
import scanet.models.Aggregation.Avg
import scanet.models.layer.StatelessLayer

import scala.collection.immutable.Seq

object Math {

  case object `x^2` extends StatelessLayer {

    override def params_(input: Shape): Params[ParamDef] =
      Params(Weights -> ParamDef(Shape(), Initializer.Zeros, Some(Avg), trainable = true))

    override def buildStateless_[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] =
      pow(params(Weights), 2)

    override def penalty_[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] =
      zeros[E](Shape())

    override def outputShape(input: Shape): Shape = input
  }
}
