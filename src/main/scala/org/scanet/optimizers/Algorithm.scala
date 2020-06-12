package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor}

trait Algorithm {

  def initMeta(shape: Shape): Tensor[Float]

  def delta(grad: Output[Float], meta: Output[Float], iter: Output[Int]): Delta
}
case class Delta(delta: Output[Float], meta: Output[Float])