package scanet.core

import scanet.core.Session.withing

trait Eval[M] {
  def eval: M
  def display(dir: String = ""): Unit
}

object Eval {
  trait AllSyntax {
    implicit def canEval[A](value: A)(implicit m: Mat[A]): Eval[m.Out] =
      new Eval[m.Out] {
        override def eval: m.Out = withing { session =>
          val (layout, allExpr) = m.deconstructIn(value)
          m.constructOutRaw(layout, session.runner.evalUnsafe(allExpr))
        }
        override def display(dir: String): Unit = {
          val (_, allExpr) = m.deconstructIn(value)
          TensorBoard(dir).addGraph(allExpr: _*)
        }
      }
  }
  object syntax extends AllSyntax
}
