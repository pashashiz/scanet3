package org.scanet.core

import org.scanet.core.Session.syntax._
import org.scanet.core.Session.withing
import org.scanet.math.syntax.placeholder

trait TFu1[P1, P1M, R] {
  def build(shapes: Seq[Shape]): (Seq[Output[P1]], R)
  def compile(session: Session)(implicit res: CanEval[R]): P1M => res.Materialized
}

object TFu1 {
  def apply[A1[_], P1: TensorType, R](func: A1[P1] => R)
                                     (implicit arg1: OutputContainer[A1]): TFu1[P1, arg1.Materialized[P1], R] =
    new TFu1[P1, arg1.Materialized[P1], R] {

      private val buildMemoized = memoize(s => build(s))

      override def build(shapes: Seq[Shape]): (Seq[Output[P1]], R) = {
        val p1 = shapes.map(placeholder[P1](_))
        val out = func(arg1.of(p1))
        (p1, out)
      }

      override def compile(session: Session)
                          (implicit res: CanEval[R]): arg1.Materialized[P1] => res.Materialized = {
        in: arg1.Materialized[P1] => {
          val tensors = arg1.materializedToSeq(in)
          val (placeholders, out) = buildMemoized(tensors.map(_.shape))
          session.runner.feed(placeholders zip tensors: _*).evalU(out)(res)
        }
      }
    }
}