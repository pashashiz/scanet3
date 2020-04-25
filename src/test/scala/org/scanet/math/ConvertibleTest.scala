package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.math.syntax._

class ConvertibleTest extends AnyFlatSpec with Matchers {

  "Float" should "be converted to Int" in {
    Convertible[Float, Int].convert(2.5f) should be(2)
  }
}
