package org.scanet.optimizers

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric

abstract class Model[A1: Numeric: TensorType, A2: Numeric: TensorType, B: Numeric: TensorType] extends (Output[A1] => TensorFunction[A2, B])

object Model {
  def apply[A1: Numeric: TensorType, A2: Numeric: TensorType, B: Numeric: TensorType] (builder: Output[A1] => TensorFunction[A2, B]): Model[A1, A2, B] =
    new Model[A1, A2, B]() {
      override def apply(x: Output[A1]): TensorFunction[A2, B] = builder(x)
    }
}
