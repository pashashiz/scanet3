package org.scanet.models.nn

import org.scanet.models.Model

trait Layer extends Model {

  /**
   * Compose `right` layer with `this` (`left`) layer.
   *
   * Composing layers is like function composition,
   * the result of `left` layer will be passed to `right` layer as output.
   * After composing resulting layer will inherit all weights from both layers.
   *
   * @param right layer
   * @return composed
   */
  def >> (right: Layer): Composed = andThen(right)

  def andThen(right: Layer): Composed = Composed(this, right)
}
