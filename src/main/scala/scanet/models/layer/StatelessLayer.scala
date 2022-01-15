package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.syntax.zeros

import scala.collection.immutable.Seq

trait StatelessLayer extends Layer {
  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())
  override def weightsCount: Int = 0
  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty
  override def initWeights[E: Floating](input: Shape): OutputSeq[E] = Seq.empty
}
