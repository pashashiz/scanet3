package scanet.models.layer

import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.syntax.zeros
import scanet.models.{Model, ParamDef}

import scala.collection.immutable.Seq
import scala.annotation.nowarn

trait Layer extends Model {

  def trainable: Boolean = true
  def stateful: Boolean

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

  override def stateful: Boolean = false

  def buildStateless_[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E]

  override def build_[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    (buildStateless_(input, params), Params.empty)
  }
}

trait NotTrainableLayer extends StatelessLayer {

  override def trainable: Boolean = false

  override def params_(input: Shape): Params[ParamDef] = Params.empty
  override def penalty_[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] =
    zeros[E](Shape())

  def build_[E: Floating](input: Expr[E]): Expr[E]

  override def buildStateless_[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] = {
    require(params.isEmpty, s"$this layer does not require params")
    build_(input)
  }
}
