package org.scanet.optimizers

import org.scanet.core.{Output, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

abstract class TensorFunction[A: Numeric: TensorType, B: Numeric: TensorType] extends (Output[A] => Output[B]){
  def apply(arg: Output[A]): Output[B]
  def grad(arg: Output[A]): Output[Float] = {
    apply(arg).grad(arg)
  }
}

object TensorFunction {
  def apply[A: Numeric: TensorType, B: Numeric: TensorType] (f: Output[A] => Output[B]): TensorFunction[A, B] = new TensorFunction[A, B] {
    override def apply(arg: Output[A]): Output[B] = f(arg)
  }
}