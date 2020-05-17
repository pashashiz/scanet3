package org.scanet.core

class TF1[P1: TensorType, R: TensorType](val builder: () => (Output[P1], Output[R])) {

  def compile(session: Session): Tensor[P1] => Tensor[R] = {
    val (p1, out) = builder()
    a1 => session.runner.feed(p1 -> a1).eval(out)
  }

  def compose[P2: TensorType, OR: TensorType, CR: TensorType](other: TF1[P2, OR])(via: (Output[R], Output[OR]) => Output[CR]): TF2[P1, P2, CR] = {
    TF2[P1, P2, CR] {
      val (p1, out) = builder()
      val (p2, outOther) = other.builder()
      (p1, p2, via(out, outOther))
    }
  }

  // todo: compose TF2, TF3
}

object TF1 {
  def apply[P1: TensorType, R: TensorType](builder: => (Output[P1], Output[R])): TF1[P1, R] = new TF1(() => builder)
}

class TF2[P1: TensorType, P2: TensorType, R: TensorType](val builder: () => (Output[P1], Output[P2], Output[R])) {

  def compile(session: Session): (Tensor[P1], Tensor[P2]) => Tensor[R] = {
    val (p1, p2, out) = builder()
    (a1, a2) => session.runner.feed(p1 -> a1, p2 -> a2).eval(out)
  }

  // todo: compose TF1, TF2, TF3

}

object TF2 {
  def apply[P1: TensorType, P2: TensorType, R: TensorType](builder: => (Output[P1], Output[P2], Output[R])): TF2[P1, P2, R] = new TF2(() => builder)
}


// todo: TF3

// todo: figure out what to doo with multiple outputs...