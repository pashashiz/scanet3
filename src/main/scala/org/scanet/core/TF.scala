package org.scanet.core

import org.scanet.core.Session.withing
import org.scanet.core.syntax._

class TF1[P1: TensorType, In: SessionInput, Out: SessionOutput]
    (val builder: Shape => (Output[P1], In), val id: Option[String] = None) {

  private val buildIfAbsent: Shape => (Output[P1], In) = memoize(builder)

  def withId(id: String): TF1[P1, In, Out] = new TF1(builder, Some(id))

  def compile(session: Session): Tensor[P1] => Out = {
    a1 => {
      val (p1, out) = buildIfAbsent(a1.shape)
      session.runner.feed(p1 -> a1).evalX[In, Out](out)
    }
  }

  def compile(): Tensor[P1] => Out =
    p1 => {
      withing(session => {
        compile(session).apply(p1)
      })
    }

  def compose[P2: TensorType, InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput, OutNew: SessionOutput]
    (other: TF1[P2, InOther, OutOther])(via: (In, InOther) => InNew): TF2[P1, P2, InNew, OutNew] = {
    new TF2((a1, a2) => {
      val (p1, out) = builder(a1)
      val (p2, ouOutOther) = other.builder(a2)
      (p1, p2, via(out, ouOutOther))
    })
  }

  def compose[P1Other: TensorType, P2Other: TensorType, InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput]
  (other: TF2[P1Other, P2Other, InOther, OutOther])(via: (In, InOther) => InNew): Composition[P1Other, P2Other, InOther, OutOther, InNew] =
    new Composition(other, via)

  class Composition[P1Other: TensorType, P2Other: TensorType, InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput]
  (private val other: TF2[P1Other, P2Other, InOther, OutOther], private val via: (In, InOther) => InNew) {
    def into[OutNew: SessionOutput]: TF3[P1, P1Other, P2Other, InNew, OutNew] = {
      new TF3((a1, a2, a3) => {
        val (p1, out) = builder(a1)
        val (p2, p3, ouOutOther) = other.builder(a2, a3)
        (p1, p2, p3, via(out, ouOutOther))
      })
    }
  }
}

object TF1 {

  case class TF1Builder[P1: TensorType, In: SessionInput](builder: Output[P1] => In) {
    def returns[Out: SessionOutput]: TF1[P1, In, Out] = new TF1[P1, In, Out](shape => {
      val arg1 = placeholder[P1](shape)
      (arg1, builder(arg1))
    })
  }

  def apply[P1: TensorType, In: SessionInput](builder: Output[P1] => In): TF1Builder[P1, In] = TF1Builder(builder)

  def identity[P: TensorType]: TF1[P, Output[P], Tensor[P]] = TF1[P, Output[P]](arg => arg).returns[Tensor[P]]
}

class TF2[P1: TensorType, P2: TensorType, In: SessionInput, Out: SessionOutput]
    (val builder: (Shape, Shape) => (Output[P1], Output[P2], In)) {

  private val buildIfAbsent: (Shape, Shape) => (Output[P1], Output[P2], In) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2]) => Out = {
    (a1, a2) => {
      val (p1, p2, out) = buildIfAbsent(a1.shape, a2.shape)
      session.runner.feed(p1 -> a1, p2 -> a2).evalX[In, Out](out)
    }
  }

  def compile(): (Tensor[P1], Tensor[P2])  => Out =
    (p1, p2) => {
      withing(session => {
        compile(session).apply(p1, p2)
      })
    }

  def compose[P1Other: TensorType, InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput, OutNew: SessionOutput]
    (other: TF1[P1Other, InOther, OutOther])(via: (In, InOther) => InNew): TF3[P1, P2, P1Other, InNew, OutNew] = {
    new TF3((a1, a2, a3) => {
      val (p1, p2, out) = builder(a1, a2)
      val (p3, ouOutOther) = other.builder(a3)
      (p1, p2, p3, via(out, ouOutOther))
    })
  }

  def compose[P1Other: TensorType, P2Other: TensorType, InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput]
  (other: TF2[P1Other, P2Other, InOther, OutOther])(via: (In, InOther) => InNew): Composition[P1Other, P2Other, InOther, OutOther, InNew] =
    new Composition(other, via)

  class Composition[P1Other: TensorType, P2Other: TensorType, InOther: SessionInput, OutOther: SessionOutput, InNew: SessionInput]
  (private val other: TF2[P1Other, P2Other, InOther, OutOther], private val via: (In, InOther) => InNew) {
    def into[OutNew: SessionOutput]: TF4[P1, P2, P1Other, P2Other, InNew, OutNew] = {
      new TF4((a1, a2, a3, a4) => {
        val (p1, p2, out) = builder(a1, a2)
        val (p3, p4, ouOutOther) = other.builder(a3, a4)
        (p1, p2, p3, p4, via(out, ouOutOther))
      })
    }
  }

  def map[InNew: SessionInput, OutNew: SessionOutput](mapper: In => InNew): TF2[P1, P2, InNew, OutNew] = {
    new TF2((a1, a2) => {
      val (p1, p2, out) = builder(a1, a2)
      (p1, p2, mapper(out))
    })
  }
}

object TF2 {

  case class TF2Builder[P1: TensorType, P2: TensorType, In: SessionInput](builder: (Output[P1], Output[P2]) => In) {
    def returns[Out: SessionOutput]: TF2[P1, P2, In, Out] = new TF2[P1, P2, In, Out](
      (shape1, shape2) => {
        val arg1 = placeholder[P1](shape1)
        val arg2 = placeholder[P2](shape2)
        (arg1, arg2, builder(arg1, arg2))
      })
  }

  def apply[P1: TensorType, P2: TensorType, In: SessionInput](builder: (Output[P1], Output[P2]) => In): TF2Builder[P1, P2, In] = TF2Builder(builder)

  def identity[P1: TensorType, P2: TensorType]: TF2[P1, P2, (Output[P1], Output[P2]), (Tensor[P1], Tensor[P2])] =
    TF2[P1, P2, (Output[P1], Output[P2])]((arg1, arg2) => (arg1, arg2)).returns[(Tensor[P1], Tensor[P2])]
}

class TF3[P1: TensorType, P2: TensorType, P3: TensorType, In: SessionInput, Out: SessionOutput](
    val builder: (Shape, Shape, Shape) => (Output[P1], Output[P2], Output[P3], In)) {

  private val buildIfAbsent: (Shape, Shape, Shape) => (Output[P1], Output[P2], Output[P3], In) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2], Tensor[P3]) => Out = {
    (a1, a2, a3) => {
      val (p1, p2, p3, out) = buildIfAbsent(a1.shape, a2.shape, a3.shape)
      session.runner.feed(p1 -> a1, p2 -> a2, p3 -> a3).evalX[In, Out](out)
    }
  }

  def compile(): (Tensor[P1], Tensor[P2], Tensor[P3]) => Out =
    (p1, p2, p3) => {
      withing(session => {
        compile(session).apply(p1, p2, p3)
      })
    }

}

object TF3 {

  case class TF3Builder[P1: TensorType, P2: TensorType, P3: TensorType, In: SessionInput]
    (builder: (Output[P1], Output[P2], Output[P3]) => In) {
    def returns[Out: SessionOutput]: TF3[P1, P2, P3, In, Out] = new TF3[P1, P2, P3, In, Out](
      (shape1, shape2, shape3) => {
        val arg1 = placeholder[P1](shape1)
        val arg2 = placeholder[P2](shape2)
        val arg3 = placeholder[P3](shape3)
        (arg1, arg2, arg3, builder(arg1, arg2, arg3))
      })
  }

  def apply[P1: TensorType, P2: TensorType, P3: TensorType, In: SessionInput]
    (builder: (Output[P1], Output[P2], Output[P3]) => In): TF3Builder[P1, P2, P3, In] = TF3Builder(builder)
}

class TF4[P1: TensorType, P2: TensorType, P3: TensorType, P4: TensorType, In: SessionInput, Out: SessionOutput](
  val builder: (Shape, Shape, Shape, Shape) => (Output[P1], Output[P2], Output[P3], Output[P4], In)) {

  private val buildIfAbsent: (Shape, Shape, Shape, Shape) => (Output[P1], Output[P2], Output[P3], Output[P4], In) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2], Tensor[P3], Tensor[P4]) => Out = {
    (a1, a2, a3, a4) => {
      val (p1, p2, p3, p4, out) = buildIfAbsent(a1.shape, a2.shape, a3.shape, a4.shape)
      session.runner.feed(p1 -> a1, p2 -> a2, p3 -> a3, p4 -> a4).evalX[In, Out](out)
    }
  }

  def compile(): (Tensor[P1], Tensor[P2], Tensor[P3], Tensor[P4]) => Out =
    (p1, p2, p3, p4) => {
      withing(session => {
        compile(session).apply(p1, p2, p3, p4)
      })
    }
}

object TF4 {

  case class TF4Builder[P1: TensorType, P2: TensorType, P3: TensorType, P4: TensorType, In: SessionInput]
  (builder: (Output[P1], Output[P2], Output[P3], Output[P4]) => In) {
    def returns[Out: SessionOutput]: TF4[P1, P2, P3, P4, In, Out] = new TF4[P1, P2, P3, P4, In, Out](
      (shape1, shape2, shape3, shape4) => {
        val arg1 = placeholder[P1](shape1)
        val arg2 = placeholder[P2](shape2)
        val arg3 = placeholder[P3](shape3)
        val arg4 = placeholder[P4](shape4)
        (arg1, arg2, arg3, arg4, builder(arg1, arg2, arg3, arg4))
      })
  }

  def apply[P1: TensorType, P2: TensorType, P3: TensorType, P4: TensorType, In: SessionInput]
  (builder: (Output[P1], Output[P2], Output[P3], Output[P4]) => In): TF4Builder[P1, P2, P3, P4, In] = TF4Builder(builder)
}