package org.scanet.core

import org.scanet.core.Session.syntaxX._
import org.scanet.core.Session.withing
import org.scanet.math.syntax.placeholder

class TFx1[A1[_]: SeqLike, P1: TensorType, In: SessionInputX, Out: SessionOutputX]
(val builder: A1[Shape] => (Seq[Output[P1]], In)) {

  private val buildIfAbsent: A1[Shape] => (Seq[Output[P1]], In) = memoize(builder)

  def compile(session: Session): A1[Tensor[P1]] => Out = {
    a1 => {
      val tensors = a1.asSeq
      val (p1, out) = buildIfAbsent(SeqLike[A1].unit(tensors.map(_.shape)))
      session.runner.feed(p1 zip tensors:_*).evalXX[In, Out](out)
    }
  }

  def compile(): A1[Tensor[P1]] => Out =
    a1 => {
      withing(session => {
        compile(session).apply(a1)
      })
    }
}

object TFx1 {

  case class TFx1Builder[A1[_]: SeqLike, P1: TensorType, In: SessionInputX]
  (builder: A1[Output[P1]] => In) {
    def returns[Out: SessionOutputX]: TFx1[A1, P1, In, Out] = new TFx1[A1, P1, In, Out](shapes => {
      val placeholders = shapes.asSeq.map(shape => placeholder[P1](shape))
      val out = builder(SeqLike[A1].unit(placeholders))
      (placeholders, out)
    })
  }

  def apply[A1[_]: SeqLike, P1: TensorType, In: SessionInputX]
  (builder: A1[Output[P1]] => In): TFx1Builder[A1, P1, In] =
    TFx1Builder(builder)

  def identity[A1[_]: SeqLike, P: TensorType]: TFx1[A1, P, A1[Output[P]], A1[Tensor[P]]] =
    TFx1[A1, P, A1[Output[P]]](arg => arg).returns[A1[Tensor[P]]]
}
