package scanet.models

import scanet.core.{Expr, Floating, Shape}
import scanet.math.syntax._
import scanet.models.layer.StatelessLayer

import scala.collection.immutable.Seq

object Math {

  case object `x^2` extends StatelessLayer {

    override def build[A: Floating](
        input: Expr[A],
        weights: Seq[Expr[A]]): Expr[A] =
      weights.head * weights.head

    override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]) = zeros[E](Shape())

    override def weightsShapes(input: Shape): Seq[Shape] = Seq(Shape())

    override def outputShape(input: Shape): Shape = input

    override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] = Seq(zeros[E](Shape()))
  }
}
