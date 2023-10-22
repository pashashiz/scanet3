package scanet.models

import scanet.core.{Expr, Floating}
import scanet.math.syntax._
import scala.collection.immutable.Seq

trait Aggregation {
  def build[E: Floating](inputs: Seq[Expr[E]]): Expr[E]
}

object Aggregation {
  case object Sum extends Aggregation {
    override def build[E: Floating](inputs: Seq[Expr[E]]): Expr[E] =
      plus(inputs)
  }
  case object Avg extends Aggregation {
    override def build[E: Floating](inputs: Seq[Expr[E]]): Expr[E] =
      plus(inputs) / inputs.size.const.cast[E]
  }
}
