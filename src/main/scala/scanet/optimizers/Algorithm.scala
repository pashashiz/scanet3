package scanet.optimizers

import scanet.core.{Expr, Floating, Params, Shape}
import scanet.models.ParamDef

trait Algorithm {
  def params(input: Shape): Params[ParamDef]
  def build[T: Floating](grad: Expr[T], params: Params[Expr[T]], iter: Expr[Int]): Delta[T]
}

case class Delta[T: Floating](value: Expr[T], params: Params[Expr[T]])
