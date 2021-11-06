package scanet.core

object Require {

  class RequireOps[A: TensorType](expr: Expr[A]) {
    def requireSquareMatrixTail: Expr[A] = {
      require(
        expr.rank >= 2,
        s"at least tensor with rank 2 is required but was passed a tensor with rank ${expr.rank}")
      val matrix = Shape(expr.shape.dims.takeRight(2))
      require(
        matrix.dims.distinct.size <= 1,
        s"the last 2 dimensions should form a squared matrix, but was a matrix with shape $matrix")
      expr
    }
  }

  trait AllSyntax {
    implicit def toRequireOps[A: TensorType](expr: Expr[A]): RequireOps[A] = new RequireOps(expr)
  }

  object syntax extends AllSyntax
}
