package scanet.models.layer

import scanet.models.Model

import scala.annotation.nowarn

trait Layer extends Model {

  def weightsCount: Int = 1
  def trainable: Boolean = weightsCount > 0

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
