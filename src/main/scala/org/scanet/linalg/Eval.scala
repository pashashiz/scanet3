package org.scanet.linalg

import org.scanet.core.Numeric
import org.tensorflow.{Tensor => NativeTensor}

// Op[A] => Tensor[A]
// (Op[A], Op[B]) => (Tensor[A], Tensor[B])
// (Op[A], Op[B], Op[C]) => ...

// todo: see how shapeless handles that, for now use simple way without typeclasses

case class OpEval[A: Numeric](op: Op[A]) {
  def eval: Tensor[A] = {
    Session.run(op)
  }
}

case class Tuple2Eval[A1: Numeric, A2: Numeric](tuple2: (Op[A1], Op[A2])) {
  def eval: (Tensor[A1], Tensor[A2]) = {
    val tensors = Session.runN(List(tuple2._1, tuple2._2))
    (Tensor[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
      Tensor[A2](tensors(1).asInstanceOf[NativeTensor[A2]]))
  }
}

case class Tuple3Eval[A1: Numeric, A2: Numeric, A3: Numeric](tuple: (Op[A1], Op[A2], Op[A3])) {
  def eval: (Tensor[A1], Tensor[A2], Tensor[A3]) = {
    val tensors = Session.runN(List(tuple._1, tuple._2, tuple._3))
    (Tensor[A1](tensors(0).asInstanceOf[NativeTensor[A1]]),
      Tensor[A2](tensors(1).asInstanceOf[NativeTensor[A2]]),
      Tensor[A3](tensors(2).asInstanceOf[NativeTensor[A3]])
    )
  }
}

object Eval {
  trait Syntax {
    implicit def tuple2Eval[A: Numeric, B: Numeric](tuple: (Op[A], Op[B])): Tuple2Eval[A, B] = Tuple2Eval(tuple)
    implicit def opEval[A: Numeric](op: Op[A]): OpEval[A] = OpEval(op)
  }
  object syntax extends Syntax
}

