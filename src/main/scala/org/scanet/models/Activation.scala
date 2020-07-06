package org.scanet.models

import org.scanet.core.{Output, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

trait Activation {
  def build[A: Numeric: Floating: TensorType](in: Output[A]): Output[A]
}

object Sigmoid extends Activation {
  override def build[A: Numeric : Floating : TensorType](in: Output[A]): Output[A] = in.sigmoid
}
