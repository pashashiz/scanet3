package scanet.models

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.syntax._

import scala.collection.immutable.Seq

object Math {

  case object `x^2` extends Model {

    override def build[A: Floating](
        x: Expr[A],
        weights: OutputSeq[A]): Expr[A] =
      weights.head * weights.head

    override def penalty[E: Floating](weights: OutputSeq[E]) =
      zeros[E](Shape())

    override def weightsShapes(input: Shape): Seq[Shape] = Seq(Shape())

    override def outputShape(input: Shape): Shape = Shape(1)

    override def initWeights[E: Floating](input: Shape): OutputSeq[E] = Seq(zeros[E](Shape()))
  }
}
