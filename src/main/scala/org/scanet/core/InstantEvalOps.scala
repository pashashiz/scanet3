package org.scanet.core

import org.scanet.core.Session.withing

trait InstantEvalOps[M] {
  def eval: M
}

object InstantEvalOps {
  trait Syntax {
    implicit def canInstantlyEval[A](out: A)(implicit ce: CanEval[A]): InstantEvalOps[ce.Materialized] =
      new InstantEvalOps[ce.Materialized] {
        override def eval: ce.Materialized = withing(session => ce.eval(session.runner, out))
      }
  }
  object syntax extends Syntax
}

// todo: add tensor board ops