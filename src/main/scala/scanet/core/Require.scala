package scanet.core

object Require {

  def fail(error: String): Nothing =
    throw new IllegalArgumentException(error)

  private def nonEmptyJoin(words: String*) = words.filter(_.nonEmpty).mkString(" ")

  class RequireOps[A: TensorType](expr: Expr[A]) {

    def requireRank(rank: Int, as: String = ""): Expr[A] = {
      require(
        expr.rank == rank,
        nonEmptyJoin(as, s"tensor with rank=$rank is required but was rank=${expr.rank}"))
      expr
    }

    def requireAtLestRank(rank: Int, as: String = ""): Expr[A] = {
      require(
        expr.rank >= rank,
        nonEmptyJoin(as, s"tensor with rank=>$rank is required but was rank=${expr.rank}"))
      expr
    }

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
