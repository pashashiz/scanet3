package org.scanet.optimizers

import org.scanet.core.{Output, Shape, Tensor, TensorType}
import org.scanet.math.{Floating, Numeric}

trait Algorithm {

  def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T]

  def delta[T: Floating: Numeric: TensorType](grad: Output[T], meta: Output[T], iter: Output[Int]): Delta[T]
}
case class Delta[T: Floating: Numeric: TensorType](delta: Output[T], meta: Output[T])