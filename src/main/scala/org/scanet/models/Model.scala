package org.scanet.models

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric

abstract class Model[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType]
  extends ((Output[X], Output[W]) => Output[J]) {
}

object Model {
  def apply[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType] (
            builder: (Output[X], Output[W]) => Output[J]): Model[X, W, J] =
    new Model[X, W, J]() {
      override def apply(batch: Output[X], weights: Output[W]): Output[J] = builder(batch, weights)
    }
}
