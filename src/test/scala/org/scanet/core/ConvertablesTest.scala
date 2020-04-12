package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.syntax.core._

class ConvertablesTest extends AnyFlatSpec with Matchers {

  "Float" should "be converted to Int" in {
    ConvertableTo[Int].fromFloat(2.5f) should be(2)
  }
}
