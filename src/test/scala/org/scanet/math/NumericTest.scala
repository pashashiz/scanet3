package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.math.syntax._

class NumericTest extends AnyFlatSpec with Matchers {

  "division" should "work for same types" in {
    10.div(2) should be(5)
  }

  "division" should "work for different types" in {
    def divTypes[A: Numeric](a: A, b: Int)(implicit c: Convertible[Int, A]): A = a.div(b)
    divTypes(10.0f, 2) should be(5.0f)
  }

  "plus op" should "work on numeric types" in {
    // NOTE: + op does no work
    def checkOrder2[A: Numeric, B: Numeric](a: A, b: B)(implicit c: Convertible[B, A]): A = a plus b
    checkOrder2(1, 2.0f) should be(3)
  }
}
