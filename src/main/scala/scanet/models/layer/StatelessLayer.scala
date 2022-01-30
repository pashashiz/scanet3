package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.math.syntax.zeros

import scala.collection.immutable.Seq

trait StatelessLayer extends Layer {
  override def penalty[E: Floating](weights: Seq[Expr[E]]): Expr[E] = zeros[E](Shape())
  override def weightsCount: Int = 0
  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty
  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] = Seq.empty
}
