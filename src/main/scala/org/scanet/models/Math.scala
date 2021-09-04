package org.scanet.models

import org.scanet.core.{Expr, OutputSeq, Shape, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._
import scala.collection.immutable.Seq

object Math {

  case object `x^2` extends Model {

    override def build[A: Numeric: Floating: TensorType](
        x: Expr[A],
        weights: OutputSeq[A]): Expr[A] =
      weights.head * weights.head

    override def penalty[E: Numeric: Floating: TensorType](weights: OutputSeq[E]) =
      zeros[E](Shape())

    override def shapes(features: Int): Seq[Shape] = Seq(Shape())

    override def outputs(): Int = 1
  }
}
