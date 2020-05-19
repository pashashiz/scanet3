package org.scanet.core

case class TFX1[P1: TensorType, O, T](builder: Shape => (Output[P1], O)) {

  private val buildIfAbsent: Shape => (Output[P1], O) = memoize(builder)

  def compile(session: Session): Tensor[P1] => T = {
    a1 => {
      val (p1, out) = buildIfAbsent(a1.shape)
      // val outputs = out.convert | O -> Seq[Output[_]
      // val native = evalUnsafe(outputs)
      // native.convert | Seq[NativeTensor[_]] -> T

      // session.runner.feed(p1 -> a1).eval(out)
      ???
    }
  }

  def compose[P2: TensorType, O_OTHER, O_NEW, T_NEW](other: TF1[P2, O_OTHER])(via: (O, O_OTHER) => O_NEW): TFX2[P1, P2, O_NEW, T_NEW] = {
    TFX2[P1, P2, O_NEW, T_NEW]((a1, a2) => {
      val (p1, out) = builder(a1)
      val (p2, outOther) = other.builder(a2)
      (p1, p2, via(out, outOther))
    })
  }

  // todo: compose TF2, TF3
}

case class TFX2[P1: TensorType, P2: TensorType, O, T](builder: (Shape, Shape) => (Output[P1], Output[P2], O)) {

  private val buildIfAbsent: (Shape, Shape) => (Output[P1], Output[P2], O) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2]) => T = {
    (a1, a2) => {
      val (p1, p2, out) = buildIfAbsent(a1.shape, a2.shape)
      // session.runner.feed(p1 -> a1, p2 -> a2).eval(out)
      ???
    }
  }

  // todo: compose TF1, TF2, TF3

}

// todo: TF3

// todo: figure out what to doo with multiple outputs...