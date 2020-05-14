package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._
import org.scanet.models.TensorFunction

case class SGD(rate: Double = 0.01, momentum: Double = 0.0, nesterov: Boolean = false) extends Algorithm[Tensor[Float]] {

  override def delta[A: Numeric : TensorType, B: Numeric : TensorType](f: TensorFunction[A, B], initArg: Output[A], prevState: Tensor[Float]): (Output[Float], Tensor[Float]) = {
    val g = f.grad(initArg)
    val m = momentum.toFloat.const
    val r = rate.toFloat.const
    val v = r * g - m * prevState.const
    val delta = if (nesterov) r * g - m * v else v
    (delta, v.eval)
  }

  override def initState[A: Numeric : TensorType](initArg: Tensor[A]): Tensor[Float] = Tensor.zeros[Float](initArg.shape)
}
