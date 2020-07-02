package org.scanet.models

import org.scanet.core.{Output, Shape, TensorType, error}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

object Math {

  case class `x^2`[E: Floating: Numeric: TensorType]() extends Model[E, E] {

    override def buildResult(x: Output[E], weights: Output[E]): Output[E] =
      error("result function is not supported for pure math models")

    override def buildLoss(x : Output[E], y: Output[E], weights: Output[E]): Output[E] =
      weights * weights

    override def weightsShape(features: Int): Shape = Shape()

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
