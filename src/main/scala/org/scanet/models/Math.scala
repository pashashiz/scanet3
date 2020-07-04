package org.scanet.models

import org.scanet.core.{Output, Shape, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

object Math {

  case object `x^2` extends Model {

    override def build[A: Numeric: Floating: TensorType](x: Output[A], weights: Output[A]): Output[A] =
      weights * weights

    override def shape(features: Int): Shape = Shape()

    override def outputs(): Int = 1
  }


//  // for each dataset row we will calcullate:
//  // x0 * w0^0 + x1 * w1^1 + x2 * w1^2 + ... + xn * wn^n
//  // after all those records will be summed
//  // usually to test a simple polynomial we need to have one row in a detaset
//  def polynomial(exponent: Int): Model[Float, Float, Float] = Model[Float, Float, Float](
//    "polynomial",
//    (x, w) => {
//      ??? // todo: need slice and concat operations
//    }, features => Shape(features))

}
