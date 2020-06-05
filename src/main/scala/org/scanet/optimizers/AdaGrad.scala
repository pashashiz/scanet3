package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

/**
 * Adagrad is an optimizer with parameter-specific learning rates,
 * which are adapted relative to how frequently a parameter gets updated during training.
 * The more updates a parameter receives, the smaller the updates.
 *
 * @param rate the learning rate.
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class AdaGrad(rate: Double = 1.0, epsilon: Double = 1e-7) extends Algorithm {

  def initMeta(shape: Shape): Tensor[Float] = Tensor.zeros[Float](shape)

  def delta(grad: Output[Float], prevGradAcc: Output[Float]): Delta = {
    // we accumulate all squared gradient per each weight
    val gradAcc = prevGradAcc + grad.sqr
    // the larger gradient is accumulated the lower rate is applied for a given weight
    val rates = rate.toFloat.const / (gradAcc.sqrt + epsilon.toFloat.const)
    val delta = rates :* grad
    Delta(delta, gradAcc)
  }
}
