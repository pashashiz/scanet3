package org.scanet.models

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric

abstract class Model[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType] extends (Output[X] => TensorFunction[W, J])

object Model {
  def apply[X: Numeric: TensorType, W: Numeric: TensorType, J: Numeric: TensorType] (
            builder: Output[X] => TensorFunction[W, J]): Model[X, W, J] =
    new Model[X, W, J]() {
      override def apply(x: Output[X]): TensorFunction[W, J] = builder(x)
    }
}
