package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.math.Numeric

trait Algorithm[S] {

  def delta[A: Numeric : TensorType, B: Numeric : TensorType](f: TensorFunction[A, B], initArg: Output[A], prevState: S): (Output[Float], S)

  def initState[A: Numeric : TensorType](initArg: Tensor[A]): S
}