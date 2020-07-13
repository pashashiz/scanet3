package org.scanet.core

import org.scanet.core.Session.withing

trait InstantEvalOps[M] {
  def eval: M
  def display(dir: String = ""): Unit
}

object InstantEvalOps {
  trait Syntax {
    implicit def canInstantlyEval[A](out: A)(implicit ce: CanEval[A]): InstantEvalOps[ce.Materialized] =
      new InstantEvalOps[ce.Materialized] {
        override def eval: ce.Materialized = withing(session => ce.eval(session.runner, out))
        override def display(dir: String): Unit = new TensorBoard(dir).addGraph(ce.unwrap(out): _*)
      }
  }
  object syntax extends Syntax
}