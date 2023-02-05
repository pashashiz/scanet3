package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.math.syntax.zeros
import scanet.models.Model

import scala.collection.immutable.Seq

import scala.annotation.nowarn

trait Layer extends Model {

  def trainable: Boolean = true

  /** Compose `right` layer with `this` (`left`) layer.
    *
    * Composing layers is like function composition,
    * the result of `left` layer will be passed to `right` layer as output.
    * After composing resulting layer will inherit all weights from both layers.
    *
    * @param right layer
    * @return composed
    */
  def >>(right: Layer): Composed = andThen(right)

  def andThen(right: Layer): Composed = Composed(this, right)

  @nowarn def ?>>(cond: Boolean, right: => Layer): Layer =
    if (cond) this >> right else this
}

trait StatelessLayer extends Layer {

  def build[E: Floating](input: Expr[E], weights: Seq[Expr[E]]): Expr[E]

  override def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) =
    (build(input, weights), Seq.empty)

  override def stateShapes(input: Shape): Seq[Shape] = Seq.empty
}

trait WeightlessLayer extends StatelessLayer {
  override def trainable: Boolean = false
  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E] =
    zeros[E](Shape())
  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty
  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] = Seq.empty
}
