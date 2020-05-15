package org.scanet.optimizers

import org.scanet.core.Output
import org.scanet.math.syntax._

case class SGD(rate: Double = 0.01, momentum: Double = 0.0, nesterov: Boolean = false) extends Algorithm {

  override def delta(grad: Output[Float]): Output[Float] = {
//    val m = momentum.toFloat.const
//    val r = rate.toFloat.const
//    val v = r * grad - m * prevState.const
//    val delta = if (nesterov) r * g - m * v else v
//    (delta, v.eval)
    grad * rate.toFloat.const
  }
}
