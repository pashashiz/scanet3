package org.scanet.core

case class TF1[P1: TensorType, R: TensorType](builder: Shape => (Output[P1], Output[R])) {

  private val buildIfAbsent: Shape => (Output[P1], Output[R]) = memoize(builder)

  def compile(session: Session): Tensor[P1] => Tensor[R] = {
    a1 => {
      val (p1, out) = buildIfAbsent(a1.shape)
      session.runner.feed(p1 -> a1).eval(out)
    }
  }

  def compose[P2: TensorType, OR: TensorType, CR: TensorType](other: TF1[P2, OR])(via: (Output[R], Output[OR]) => Output[CR]): TF2[P1, P2, CR] = {
    TF2[P1, P2, CR]((a1, a2) => {
      val (p1, out) = builder(a1)
      val (p2, outOther) = other.builder(a2)
      (p1, p2, via(out, outOther))
    })
  }

  // todo: compose TF2, TF3
}

case class TF2[P1: TensorType, P2: TensorType, R: TensorType](builder: (Shape, Shape) => (Output[P1], Output[P2], Output[R])) {

  private val buildIfAbsent: (Shape, Shape) => (Output[P1], Output[P2], Output[R]) = memoize(builder)

  def compile(session: Session): (Tensor[P1], Tensor[P2]) => Tensor[R] = {
    (a1, a2) => {
      val (p1, p2, out) = buildIfAbsent(a1.shape, a2.shape)
      session.runner.feed(p1 -> a1, p2 -> a2).eval(out)
    }
  }

  // todo: compose TF1, TF2, TF3

}

// todo: TF3

// todo: figure out what to doo with multiple outputs...