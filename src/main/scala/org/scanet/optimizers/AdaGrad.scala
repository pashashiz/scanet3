package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

case class AdaGrad(rate: Double = 1) extends Algorithm {

  def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros[Float](shape)

  def delta(grad: Output[Float], prevGradAcc: Output[Float]): Delta = {
    // we accumulate all squared gradient per each weight
    val gradAcc = prevGradAcc + grad.sqr
    // the larger gradient is accumulated the lower rate is applied for a given weight
    val rates = rate.toFloat.const / (gradAcc.sqrt + 1e-7f.const)
    val delta = rates :* grad
    Delta(delta, gradAcc)
  }
}
