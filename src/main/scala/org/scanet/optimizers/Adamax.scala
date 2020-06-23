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
case class Adamax(rate: Float = 0.001f, beta1: Float = 0.9f, beta2 : Float = 0.999f, epsilon: Float = 1e-7f) extends Algorithm {

  override def initMeta(shape: Shape): Tensor[Float] = {
    val m1 = Tensor.zeros[Float](shape).const
    val m2 = Tensor.zeros[Float](shape).const
    (m1 zip m2).eval
  }

  override def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta = {
    val (prevM1, prevM2) = meta.unzip
    val m1 = prevM1.decayingAvg(grad, beta1.const)
    val m2 = max(beta2.const * prevM2, (1f.const - beta2.const) * grad.abs)
    val delta = rate.const * m1.boost(beta1.const, iter) / (m2 + epsilon.const)
    Delta(delta, m1 zip m2)
  }
}