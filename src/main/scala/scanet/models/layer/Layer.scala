package scanet.models.layer

import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.syntax.zeros
import scanet.models.{Model, ParamDef}

import scala.annotation.nowarn

trait Layer extends Model {

  def stateful: Boolean

  override def makeTrainable(trainable: Boolean): Layer

  /** Compose `right` layer with `this` (`left`) layer.
    *
    * Composing layers is like function composition,
    * the result of `left` layer will be passed to `right` layer as output.
    * After composing resulting layer will inherit all weights from both layers.
    *
    * @param right layer
    * @return composed
    */
  def >>(right: Layer): Layer = andThen(right)

  def andThen(right: Layer): Layer = Composed(this, right)

  @nowarn def ?>>(cond: Boolean, right: => Layer): Layer =
    if (cond) this >> right else this
}

trait StatelessLayer extends Layer {

  override def stateful: Boolean = false

  def buildStateless[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E]

  override def build[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    (buildStateless(input, params), Params.empty)
  }
}

trait NotTrainableLayer extends StatelessLayer {

  override def trainable: Boolean = false
  override def makeTrainable(trainable: Boolean): Layer = this

  override def params(input: Shape): Params[ParamDef] = Params.empty
  override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] =
    zeros[E](Shape())

  def build[E: Floating](input: Expr[E]): Expr[E]

  override def buildStateless[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] = {
    require(params.isEmpty, s"$this layer does not require params")
    build(input)
  }
}
