package org.scanet.optimizers

import org.scanet.core.{Expr, Shape, Tensor, TensorType}
import org.scanet.math.{Floating, Numeric}

trait Algorithm {

  def initMeta[T: Floating: Numeric: TensorType](shape: Shape): Tensor[T]

  def delta[T: Floating: Numeric: TensorType](
      grad: Expr[T],
      meta: Expr[T],
      iter: Expr[Int]): Delta[T]
}

case class Delta[T: Floating: Numeric: TensorType](delta: Expr[T], meta: Expr[T])
