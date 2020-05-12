package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

case class SGD(rate: Double = 0.01, momentum: Double = 0.0, nesterov: Boolean = false) extends Algorithm[Tensor[Float]] {

  override def delta[A: Numeric : TensorType, B: Numeric : TensorType](f: TensorFunction[A, B], initArg: Output[A], prevState: Tensor[Float]): (Output[Float], Tensor[Float]) = {
    // op = +|-
    // Nesterov == true:
    //   v[i] = momentum * v[i-1] + rate * grad(arg op momentum * v[i-1])
    // Nesterov == false:
    //   v[i] = momentum * v[i-1] + rate * grad(arg)
    // momentum == 0.0 (vanilla SGD):
    //   v[i] = rate * grad(arg)
    // arg = arg op v[i]

    val m = momentum.toFloat.const
    // TODO: should this be + for maximize?
    val arg = if (nesterov) (initArg.cast[Float] - m * prevState.const).cast[A] else initArg
    val velocity = (m * prevState.const + rate.toFloat.const * f.grad(arg)).eval
    (velocity.const, velocity)
  }

  override def initState[A: Numeric : TensorType](initArg: Tensor[A]): Tensor[Float] = Tensor.zeros[Float](initArg.shape)
}
