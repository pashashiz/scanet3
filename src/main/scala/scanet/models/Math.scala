package scanet.models

import scanet.core.Params.Weights
import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.syntax._
import scanet.models.Aggregation.Avg
import scanet.models.layer.{Layer, StatelessLayer}

object Math {

  case class `x^2`(override val trainable: Boolean = true) extends StatelessLayer {

    override def params(input: Shape): Params[ParamDef] =
      Params(Weights -> ParamDef(Shape(), Initializer.Zeros, Some(Avg), trainable = true))

    override def buildStateless[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] =
      pow(params(Weights), 2)

    override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] =
      zeros[E](Shape())

    override def outputShape(input: Shape): Shape = input

    override def makeTrainable(trainable: Boolean): Layer = copy(trainable = trainable)
  }
}
