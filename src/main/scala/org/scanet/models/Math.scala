package org.scanet.models

import org.scanet.core.{Output, Shape}
import org.scanet.math.syntax._

object Math {

  // for each dataset row we will calcullate:
  // x0 * w0^0 + x1 * w1^1 + x2 * w1^2 + ... + xn * wn^n
  // after all those records will be summed
  // usually to test a simple polynomial we need to have one row in a detaset
  def polynomial(exponent: Int): Model[Float, Float, Float] = {
    Model[Float, Float, Float]((x, w) => {
      // todo: need slice and concat operations
      ???
    }, features => Shape(features))
  }

  def `x^2`: Model[Float, Float, Float] = Model(
    (_: Output[Float], x: Output[Float]) => x * x, features => Shape(features)
  )
}
