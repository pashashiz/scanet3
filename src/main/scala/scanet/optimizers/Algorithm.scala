package scanet.optimizers

import scanet.core.{Expr, Floating, Shape, Tensor}

trait Algorithm {

  // todo: Expr[T]
  def initMeta[T: Floating](shape: Shape): Tensor[T]

  def delta[T: Floating](grad: Expr[T], meta: Expr[T], iter: Expr[Int]): Delta[T]
}

case class Delta[T: Floating](delta: Expr[T], meta: Expr[T])
