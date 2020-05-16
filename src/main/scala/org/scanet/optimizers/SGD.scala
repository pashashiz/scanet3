package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

case class SGD(rate: Double = 0.01, momentum: Double = 0.0, nesterov: Boolean = false) extends Algorithm {

  override def delta[W: TensorType : Numeric, R: TensorType : Numeric](model: Output[R], arg: Output[W]): Delta = {
    val grad = model.grad(arg)
    val r = rate.toFloat.const

    if (momentum == 0) {
      Delta(r * grad)
    } else {
      val velocity = Variable.init(Tensor.zeros[Float](arg.shape))
      val m = momentum.toFloat.const
      val v = r * grad - m * velocity.placeholder
      val delta = if (nesterov) r * grad - m * v else v
      Delta(delta, velocity.output(v))
    }
  }
}