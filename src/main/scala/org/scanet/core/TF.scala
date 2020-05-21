package org.scanet.core

import org.scanet.core.Session.using
import org.scanet.core.syntax._

class TF1[P1: TensorType, O: SessionInput, T: SessionOutput]
    (val builder: Shape => (Output[P1], O)) {

  private val buildIfAbsent: Shape => (Output[P1], O) = memoize(builder)

  def compile(session: Session): Tensor[P1] => T = {
    a1 => {
      val (p1, out: O) = buildIfAbsent(a1.shape)
      session.runner.feed(p1 -> a1).evalX[O, T](out)
    }
  }

  def compile(): Tensor[P1] => T =
    p1 => {
      using(session => {
        compile(session).apply(p1)
      })
    }

  def compose[P2: TensorType, O_OTHER: SessionInput, T_OTHER: SessionOutput, O_NEW: SessionInput, T_NEW: SessionOutput]
    (other: TF1[P2, O_OTHER, T_OTHER])(via: (O, O_OTHER) => O_NEW): TF2[P1, P2, O_NEW, T_NEW] = {
    new TF2((a1, a2) => {
      val (p1, out) = builder(a1)
      val (p2, outOther) = other.builder(a2)
      (p1, p2, via(out, outOther))
    })
  }

  def compose[P1_OTHER: TensorType, P2_OTHER: TensorType, O_OTHER: SessionInput, T_OTHER: SessionOutput, O_NEW: SessionInput, T_NEW: SessionOutput]
    (other: TF2[P1_OTHER, P2_OTHER, O_OTHER, T_OTHER])(via: (O, O_OTHER) => O_NEW): TF3[P1, P1_OTHER, P2_OTHER, O_NEW, T_NEW] = {
    new TF3((a1, a2, a3) => {
      val (p1, out) = builder(a1)
      val (p2, p3, outOther) = other.builder(a2, a3)
      (p1, p2, p3, via(out, outOther))
    })
  }
}

object TF1 {

  case class TF1Builder[P1: TensorType, O: SessionInput](builder: Output[P1] => O) {
    def returns[T: SessionOutput]: TF1[P1, O, T] = new TF1[P1, O, T](shape => {
      val arg1 = placeholder[P1](shape)
      (arg1, builder(arg1))
    })
  }

  def apply[P1: TensorType, O: SessionInput](builder: Output[P1] => O): TF1Builder[P1, O] = TF1Builder(builder)
}

class TF2[P1: TensorType, P2: TensorType, O: SessionInput, T: SessionOutput]
    (val builder: (Shape, Shape) => (Output[P1], Output[P2], O)) {

  private val buildIfAbsent: (Shape, Shape) => (Output[P1], Output[P2], O) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2]) => T = {
    (a1, a2) => {
      val (p1, p2, out) = buildIfAbsent(a1.shape, a2.shape)
      session.runner.feed(p1 -> a1, p2 -> a2).evalX[O, T](out)
    }
  }

  def compile(): (Tensor[P1], Tensor[P2])  => T =
    (p1, p2) => {
      using(session => {
        compile(session).apply(p1, p2)
      })
    }

  def compose[P1_OTHER: TensorType, O_OTHER: SessionInput, T_OTHER: SessionOutput, O_NEW: SessionInput, T_NEW: SessionOutput]
    (other: TF1[P1_OTHER, O_OTHER, T_OTHER])(via: (O, O_OTHER) => O_NEW): TF3[P1, P2, P1_OTHER, O_NEW, T_NEW] = {
    new TF3((a1, a2, a3) => {
      val (p1, p2, out) = builder(a1, a2)
      val (p3, outOther) = other.builder(a3)
      (p1, p2, p3, via(out, outOther))
    })
  }

  def map[O_NEW: SessionInput, T_NEW: SessionOutput](mapper: O => O_NEW): TF2[P1, P2, O_NEW, T_NEW] = {
    new TF2((a1, a2) => {
      val (p1, p2, out) = builder(a1, a2)
      (p1, p2, mapper(out))
    })
  }
}

object TF2 {

  case class TF2Builder[P1: TensorType, P2: TensorType, O: SessionInput](builder: (Output[P1], Output[P2]) => O) {
    def returns[T: SessionOutput]: TF2[P1, P2, O, T] = new TF2[P1, P2, O, T](
      (shape1, shape2) => {
        val arg1 = placeholder[P1](shape1)
        val arg2 = placeholder[P2](shape2)
        (arg1, arg2, builder(arg1, arg2))
      })
  }

  def apply[P1: TensorType, P2: TensorType, O: SessionInput](builder: (Output[P1], Output[P2]) => O): TF2Builder[P1, P2, O] = TF2Builder(builder)
}

class TF3[P1: TensorType, P2: TensorType, P3: TensorType, O: SessionInput, T: SessionOutput](
    val builder: (Shape, Shape, Shape) => (Output[P1], Output[P2], Output[P3], O)) {

  private val buildIfAbsent: (Shape, Shape, Shape) => (Output[P1], Output[P2], Output[P3], O) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2], Tensor[P3]) => T = {
    (a1, a2, a3) => {
      val (p1, p2, p3, out) = buildIfAbsent(a1.shape, a2.shape, a3.shape)
      session.runner.feed(p1 -> a1, p2 -> a2, p3 -> a3).evalX[O, T](out)
    }
  }

  def compile(): (Tensor[P1], Tensor[P2], Tensor[P3])  => T =
    (p1, p2, p3) => {
      using(session => {
        compile(session).apply(p1, p2, p3)
      })
    }

}

object TF3 {

  case class TF3Builder[P1: TensorType, P2: TensorType, P3: TensorType, O: SessionInput]
    (builder: (Output[P1], Output[P2], Output[P3]) => O) {
    def returns[T: SessionOutput]: TF3[P1, P2, P3, O, T] = new TF3[P1, P2, P3, O, T](
      (shape1, shape2, shape3) => {
        val arg1 = placeholder[P1](shape1)
        val arg2 = placeholder[P2](shape2)
        val arg3 = placeholder[P3](shape3)
        (arg1, arg2, arg3, builder(arg1, arg2, arg3))
      })
  }

  def apply[P1: TensorType, P2: TensorType, P3: TensorType, O: SessionInput]
    (builder: (Output[P1], Output[P2], Output[P3]) => O): TF3Builder[P1, P2, P3, O] = TF3Builder(builder)
}
