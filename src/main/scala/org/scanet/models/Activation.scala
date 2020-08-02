package org.scanet.models

import org.scanet.core.{Output, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

trait Activation {
  def build[A: Numeric: Floating: TensorType](in: Output[A]): Output[A]
}

object Activation {

  case object Identity extends Activation {
    override def build[A: Numeric : Floating : TensorType](in: Output[A]): Output[A] = in
  }

  case object Sigmoid extends Activation {
    override def build[A: Numeric : Floating : TensorType](in: Output[A]): Output[A] = in.sigmoid
  }

  case object Softmax extends Activation {
    override def build[A: Numeric : Floating : TensorType](in: Output[A]): Output[A] = {
      val e = in.exp
      val sum = e.sum(axises = Seq(1))
      e / sum.reshape(sum.shape :+ 1)
    }
  }
}