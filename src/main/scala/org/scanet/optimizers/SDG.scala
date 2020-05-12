package org.scanet.optimizers

import org.scanet.math.Numeric
import org.scanet.math.syntax._
import org.scanet.core.{Output, TensorType}

case class SDG(rate: Double = 0.01) extends Algorithm {
  override def delta[A: Numeric : TensorType, B: Numeric : TensorType](f: TensorFunction[A, B], initArg: Output[A]): Output[Float] = {
    f.grad(initArg) * rate.toFloat.const
  }
}
