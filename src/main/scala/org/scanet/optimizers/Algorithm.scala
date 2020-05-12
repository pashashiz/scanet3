package org.scanet.optimizers

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric

trait Algorithm {

  def delta[A: Numeric: TensorType, B: Numeric: TensorType](f: TensorFunction[A, B], initArg: Output[A]): Output[Float]

}
