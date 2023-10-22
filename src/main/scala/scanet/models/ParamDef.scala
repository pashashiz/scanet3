package scanet.models

import scanet.core.{Floating, Shape}

case class ParamDef(
    shape: Shape,
    initializer: Initializer = Initializer.Zeros,
    aggregation: Option[Aggregation] = None,
    trainable: Boolean = false) {
  def nonTrainable: Boolean = !trainable
  def initialize[E: Floating] = initializer.build[E](shape)
}
