package org.scanet.models

import org.scanet.core.{Output, OutputSeq, Shape, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

object Math {

  case object `x^2` extends Model {

    override def build[A: Numeric: Floating: TensorType](x: Output[A], weights: OutputSeq[A]): Output[A] =
      weights.head * weights.head

    override def shapes(features: Int): Seq[Shape] = Seq(Shape())

    override def outputs(): Int = 1
  }
}
