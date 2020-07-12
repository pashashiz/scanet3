package org.scanet.core

import org.scanet.core.Session.syntax._
import org.scanet.core.Session.withing
import org.scanet.math.syntax.placeholder

class TF1[
  A1[_]: SeqLike, P1: TensorType,
  In: SessionInput, Out: SessionOutput]
(val builder: A1[Shape] => (Seq[Output[P1]], In)) {

  private val buildIfAbsent: A1[Shape] => (Seq[Output[P1]], In) = memoize(builder)

  def compile(session: Session): A1[Tensor[P1]] => Out = {
    a1 => {
      val t1 = a1.asSeq
      val (p1, out) = buildIfAbsent(SeqLike[A1].unit(t1.map(_.shape)))
      session.runner.feed(p1 zip t1:_*).evalX[In, Out](out)
    }
  }

  def compile(): A1[Tensor[P1]] => Out =
    a1 => {
      withing(session => {
        compile(session).apply(a1)
      })
    }
}

object TF1 {

  case class TFx1Builder[A1[_]: SeqLike, P1: TensorType, In: SessionInput]
  (builder: A1[Output[P1]] => In) {
    def returns[Out: SessionOutput]: TF1[A1, P1, In, Out] = new TF1[A1, P1, In, Out](shapes => {
      val p1 = shapes.asSeq.map(placeholder[P1](_))
      val out = builder(SeqLike[A1].unit(p1))
      (p1, out)
    })
  }

  def apply[A1[_]: SeqLike, P1: TensorType, In: SessionInput]
  (builder: A1[Output[P1]] => In): TFx1Builder[A1, P1, In] =
    TFx1Builder(builder)

  def identity[A1[_]: SeqLike, P: TensorType]: TF1[A1, P, A1[Output[P]], A1[Tensor[P]]] =
    TF1[A1, P, A1[Output[P]]](arg => arg).returns[A1[Tensor[P]]]
}

class TF2[
  A1[_]: SeqLike, P1: TensorType,
  A2[_]: SeqLike, P2: TensorType,
  In: SessionInput, Out: SessionOutput]
(val builder: (A1[Shape], A2[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], In)) {

  private val buildIfAbsent: (A1[Shape], A2[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], In) = memoize(builder)

  def compile(session: Session): (A1[Tensor[P1]], A2[Tensor[P2]]) => Out = {
    (a1, a2) => {
      val t1 = a1.asSeq
      val t2 = a2.asSeq
      val (p1, p2, out) = buildIfAbsent(
        SeqLike[A1].unit(t1.map(_.shape)),
        SeqLike[A2].unit(t2.map(_.shape)))
      session.runner.feed(p1 zip t1:_*).feed(p2 zip t2:_*).evalX[In, Out](out)
    }
  }

  def compile(): (A1[Tensor[P1]], A2[Tensor[P2]])  => Out =
    (p1, p2) => {
      withing(session => {
        compile(session).apply(p1, p2)
      })
    }
}

object TF2 {

  case class TF2xBuilder[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]]) => In) {
    def returns[Out: SessionOutput]: TF2[A1, P1, A2, P2, In, Out] = new TF2[A1, P1, A2, P2, In, Out](
      (shapes1, shapes2) => {
        val p1 = shapes1.asSeq.map(placeholder[P1](_))
        val p2 = shapes2.asSeq.map(placeholder[P2](_))
        val out = builder(SeqLike[A1].unit(p1), SeqLike[A2].unit(p2))
        (p1, p2, out)
      })
  }

  def apply[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]]) => In): TF2xBuilder[A1, P1, A2, P2, In] =
    TF2xBuilder(builder)

  def identity[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType]: TF2[A1, P1, A2, P2, (A1[Output[P1]], A2[Output[P2]]), (A1[Tensor[P1]], A2[Tensor[P2]])] =
    TF2[A1, P1, A2, P2, (A1[Output[P1]], A2[Output[P2]])]((arg1, arg2) => (arg1, arg2)).returns[(A1[Tensor[P1]], A2[Tensor[P2]])]
}

class TF3[
  A1[_]: SeqLike, P1: TensorType,
  A2[_]: SeqLike, P2: TensorType,
  A3[_]: SeqLike, P3: TensorType,
  In: SessionInput,
  Out: SessionOutput]
(val builder: (A1[Shape], A2[Shape], A3[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], Seq[Output[P3]], In)) {

  private val buildIfAbsent: (A1[Shape], A2[Shape], A3[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], Seq[Output[P3]], In) = memoize(builder)

  def compile(session: Session): (A1[Tensor[P1]], A2[Tensor[P2]], A3[Tensor[P3]]) => Out = {
    (a1, a2, a3) => {
      val (t1, t2, t3) = (a1.asSeq, a2.asSeq, a3.asSeq)
      val (p1, p2, p3, out) = buildIfAbsent(
        SeqLike[A1].unit(t1.map(_.shape)),
        SeqLike[A2].unit(t2.map(_.shape)),
        SeqLike[A3].unit(t3.map(_.shape)))
      session.runner.feed(p1 zip t1:_*).feed(p2 zip t2:_*).feed(p3 zip t3:_*).evalX[In, Out](out)
    }
  }

  def compile(): (A1[Tensor[P1]], A2[Tensor[P2]], A3[Tensor[P3]]) => Out =
    (p1, p2, p3) => {
      withing(session => {
        compile(session).apply(p1, p2, p3)
      })
    }

  def compose[
    A1Other[_]: SeqLike, P1Other: TensorType,
    A2Other[_]: SeqLike, P2Other: TensorType,
    InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput]
  (other: TF2[A1Other, P1Other, A2Other, P2Other, InOther, OutOther])
  (via: (In, InOther) => InNew): CompositionWithTF2x[A1Other, P1Other, A2Other, P2Other, InOther, OutOther, InNew] =
    new CompositionWithTF2x(other, via)

  class CompositionWithTF2x[
    A1Other[_]: SeqLike, P1Other: TensorType,
    A2Other[_]: SeqLike, P2Other: TensorType,
    InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput]
  (private val other: TF2[A1Other, P1Other, A2Other, P2Other, InOther, OutOther], private val via: (In, InOther) => InNew) {
    def into[OutNew: SessionOutput]: TF5[A1, P1, A2, P2, A3, P3, A1Other, P1Other, A2Other, P2Other, InNew, OutNew] = {
      new TF5((a1, a2, a3, a4, a5) => {
        val (p1, p2, p3, out) = builder(a1, a2, a3)
        val (p4, p5, ouOutOther) = other.builder(a4, a5)
        (p1, p2, p3, p4, p5, via(out, ouOutOther))
      })
    }
  }
}

object TF3 {

  case class TF3xBuilder[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    A3[_]: SeqLike, P3: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]], A3[Output[P3]]) => In) {
    def returns[Out: SessionOutput]: TF3[A1, P1, A2, P2, A3, P3, In, Out] = new TF3[A1, P1, A2, P2, A3, P3, In, Out](
      (shapes1, shapes2, shapes3) => {
        val p1 = shapes1.asSeq.map(placeholder[P1](_))
        val p2 = shapes2.asSeq.map(placeholder[P2](_))
        val p3 = shapes3.asSeq.map(placeholder[P3](_))
        val out = builder(SeqLike[A1].unit(p1), SeqLike[A2].unit(p2), SeqLike[A3].unit(p3))
        (p1, p2, p3, out)
      })
  }

  def apply[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    A3[_]: SeqLike, P3: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]], A3[Output[P3]]) => In): TF3xBuilder[A1, P1, A2, P2, A3, P3, In] =
    TF3xBuilder(builder)
}

class TF4[
  A1[_]: SeqLike, P1: TensorType,
  A2[_]: SeqLike, P2: TensorType,
  A3[_]: SeqLike, P3: TensorType,
  A4[_]: SeqLike, P4: TensorType,
  In: SessionInput,
  Out: SessionOutput]
(val builder: (A1[Shape], A2[Shape], A3[Shape], A4[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], Seq[Output[P3]], Seq[Output[P4]], In)) {

  private val buildIfAbsent: (A1[Shape], A2[Shape], A3[Shape], A4[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], Seq[Output[P3]], Seq[Output[P4]], In) = memoize(builder)

  def compile(session: Session): (A1[Tensor[P1]], A2[Tensor[P2]], A3[Tensor[P3]], A4[Tensor[P4]]) => Out = {
    (a1, a2, a3, a4) => {
      val (t1, t2, t3, t4) = (a1.asSeq, a2.asSeq, a3.asSeq, a4.asSeq)
      val (p1, p2, p3, p4, out) = buildIfAbsent(
        SeqLike[A1].unit(t1.map(_.shape)),
        SeqLike[A2].unit(t2.map(_.shape)),
        SeqLike[A3].unit(t3.map(_.shape)),
        SeqLike[A4].unit(t4.map(_.shape)))
      session.runner.feed(p1 zip t1:_*).feed(p2 zip t2:_*).feed(p3 zip t3:_*).feed(p4 zip t4:_*).evalX[In, Out](out)
    }
  }

  def compile(): (A1[Tensor[P1]], A2[Tensor[P2]], A3[Tensor[P3]], A4[Tensor[P4]]) => Out =
    (p1, p2, p3, p4) => {
      withing(session => {
        compile(session).apply(p1, p2, p3, p4)
      })
    }
}

object TF4 {

  case class TF4xBuilder[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    A3[_]: SeqLike, P3: TensorType,
    A4[_]: SeqLike, P4: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]], A3[Output[P3]], A4[Output[P4]]) => In) {
    def returns[Out: SessionOutput]: TF4[A1, P1, A2, P2, A3, P3, A4, P4, In, Out] = new TF4[A1, P1, A2, P2, A3, P3, A4, P4, In, Out](
      (shapes1, shapes2, shapes3, shapes4) => {
        val p1 = shapes1.asSeq.map(placeholder[P1](_))
        val p2 = shapes2.asSeq.map(placeholder[P2](_))
        val p3 = shapes3.asSeq.map(placeholder[P3](_))
        val p4 = shapes4.asSeq.map(placeholder[P4](_))
        val out = builder(SeqLike[A1].unit(p1), SeqLike[A2].unit(p2), SeqLike[A3].unit(p3), SeqLike[A4].unit(p4))
        (p1, p2, p3, p4, out)
      })
  }

  def apply[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    A3[_]: SeqLike, P3: TensorType,
    A4[_]: SeqLike, P4: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]], A3[Output[P3]], A4[Output[P4]]) => In): TF4xBuilder[A1, P1, A2, P2, A3, P3, A4, P4, In] =
    TF4xBuilder(builder)
}

class TF5[
  A1[_]: SeqLike, P1: TensorType,
  A2[_]: SeqLike, P2: TensorType,
  A3[_]: SeqLike, P3: TensorType,
  A4[_]: SeqLike, P4: TensorType,
  A5[_]: SeqLike, P5: TensorType,
  In: SessionInput,
  Out: SessionOutput]
(val builder: (A1[Shape], A2[Shape], A3[Shape], A4[Shape], A5[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], Seq[Output[P3]], Seq[Output[P4]], Seq[Output[P5]], In)) {

  private val buildIfAbsent: (A1[Shape], A2[Shape], A3[Shape], A4[Shape], A5[Shape]) => (Seq[Output[P1]], Seq[Output[P2]], Seq[Output[P3]], Seq[Output[P4]], Seq[Output[P5]], In) = memoize(builder)

  def compile(session: Session): (A1[Tensor[P1]], A2[Tensor[P2]], A3[Tensor[P3]], A4[Tensor[P4]], A5[Tensor[P5]]) => Out = {
    (a1, a2, a3, a4, a5) => {
      val (t1, t2, t3, t4, t5) = (a1.asSeq, a2.asSeq, a3.asSeq, a4.asSeq, a5.asSeq)
      val (p1, p2, p3, p4, p5, out) = buildIfAbsent(
        SeqLike[A1].unit(t1.map(_.shape)),
        SeqLike[A2].unit(t2.map(_.shape)),
        SeqLike[A3].unit(t3.map(_.shape)),
        SeqLike[A4].unit(t4.map(_.shape)),
        SeqLike[A5].unit(t5.map(_.shape)))
      session.runner.feed(p1 zip t1:_*).feed(p2 zip t2:_*).feed(p3 zip t3:_*).feed(p4 zip t4:_*).feed(p5 zip t5:_*).evalX[In, Out](out)
    }
  }

  def compile(): (A1[Tensor[P1]], A2[Tensor[P2]], A3[Tensor[P3]], A4[Tensor[P4]], A5[Tensor[P5]]) => Out =
    (p1, p2, p3, p4, p5) => {
      withing(session => {
        compile(session).apply(p1, p2, p3, p4, p5)
      })
    }
}

object TF5 {

  case class TF5xBuilder[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    A3[_]: SeqLike, P3: TensorType,
    A4[_]: SeqLike, P4: TensorType,
    A5[_]: SeqLike, P5: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]], A3[Output[P3]], A4[Output[P4]], A5[Output[P5]]) => In) {
    def returns[Out: SessionOutput]: TF5[A1, P1, A2, P2, A3, P3, A4, P4, A5, P5, In, Out] = new TF5[A1, P1, A2, P2, A3, P3, A4, P4, A5, P5, In, Out](
      (shapes1, shapes2, shapes3, shapes4, shapes5) => {
        val p1 = shapes1.asSeq.map(placeholder[P1](_))
        val p2 = shapes2.asSeq.map(placeholder[P2](_))
        val p3 = shapes3.asSeq.map(placeholder[P3](_))
        val p4 = shapes4.asSeq.map(placeholder[P4](_))
        val p5 = shapes5.asSeq.map(placeholder[P5](_))
        val out = builder(SeqLike[A1].unit(p1), SeqLike[A2].unit(p2), SeqLike[A3].unit(p3), SeqLike[A4].unit(p4), SeqLike[A5].unit(p5))
        (p1, p2, p3, p4, p5, out)
      })
  }

  def apply[
    A1[_]: SeqLike, P1: TensorType,
    A2[_]: SeqLike, P2: TensorType,
    A3[_]: SeqLike, P3: TensorType,
    A4[_]: SeqLike, P4: TensorType,
    A5[_]: SeqLike, P5: TensorType,
    In: SessionInput]
  (builder: (A1[Output[P1]], A2[Output[P2]], A3[Output[P3]], A4[Output[P4]], A5[Output[P5]]) => In): TF5xBuilder[A1, P1, A2, P2, A3, P3, A4, P4, A5, P5, In] =
    TF5xBuilder(builder)
}