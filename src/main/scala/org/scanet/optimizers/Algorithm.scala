package org.scanet.optimizers

import org.scanet.core.{Output, Tensor, TensorType}

trait Algorithm {

  def initMeta[X: TensorType](initArg: Tensor[X]): Tensor[Float]

  def delta(grad: Output[Float], meta: Output[Float]): Delta
}
case class Delta(delta: Output[Float], meta: Output[Float])