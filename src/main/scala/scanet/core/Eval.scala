package scanet.core

import scanet.core.Session.withing

trait Eval[M] {
  def
  eval: M
  def display(dir: String = ""): Unit
}

object Eval {
  trait AllSyntax {
    implicit def canEval[A](out: A)(implicit ce: CanEval[A]): Eval[ce.Materialized] =
      new Eval[ce.Materialized] {
        override def eval: ce.Materialized = withing(session => ce.eval(session.runner, out))
        override def display(dir: String): Unit = TensorBoard(dir).addGraph(ce.unwrap(out): _*)
      }
  }
  object syntax extends AllSyntax
}
