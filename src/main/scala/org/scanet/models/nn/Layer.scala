package org.scanet.models.nn

import org.scanet.models.Model

trait Layer extends Model {

  def >> (right: Layer): Composed = andThen(right)

  def andThen(right: Layer): Composed = Composed(this, right)
}
