package org.scanet.optimizers

import org.scanet.core._
import org.scanet.syntax._

/** Adamax is a variant of Adam based on the infinity norm.
 * It is sometimes superior to adam, specially in models with embeddings.
 *
 * In contrast to Adam, the sparse implementation of this algorithm only updates variable slices
 * and corresponding `m` and `v` terms when that part of the variable was used in the forward pass.
 * This means that the sparse behavior is contrast to the dense behavior (similar to some momentum
 * implementations which ignore momentum unless a variable slice was actually used).
 *
 * @param rate learning rate
 * @param beta1 The exponential decay rate for the 1st moment estimates
 * @param beta2 The exponential decay rate for the exponentially weighted infinity norm
 * @param epsilon a constant epsilon used to better conditioning the grad update
 */
case class Adamax(rate: Output[Float], beta1: Output[Float], beta2: Output[Float], epsilon: Output[Float]) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val m1 = Tensor.zeros[Float](shape).const
    val m2 = Tensor.zeros[Float](shape).const
    (m1 zip m2).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM1, prevM2) = meta.unzip
    val m1 = prevM1.decayingAvg(grad, beta1)
    val m2 = max(beta2 * prevM2, (1f.const - beta2) * grad.abs)
    val delta = rate * m1.boost(beta1, iter) / (m2 + epsilon)
    Delta(delta, m1 zip m2)
  }
}
object Adamax {

  def apply(rate: Double = 0.001, beta1: Double = 0.9, beta2 : Double = 0.999, epsilon: Double = 1e-7): Adamax =
    new Adamax(rate.toFloat.const, beta1.toFloat.const, beta2.toFloat.const, epsilon.toFloat.const)
}