package org.scanet.models

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class TensorFunction[W: Numeric: TensorType, J: Numeric: TensorType] extends (Output[W] => Output[J]){
  def apply(arg: Output[W]): Output[J]
  def grad(arg: Output[W]): Output[Float] = {
    apply(arg).grad(arg)
  }
}

object TensorFunction {
  def apply[W: Numeric: TensorType, J: Numeric: TensorType] (f: Output[W] => Output[J]): TensorFunction[W, J] = new TensorFunction[W, J] {
    override def apply(arg: Output[W]): Output[J] = f(arg)
  }
}