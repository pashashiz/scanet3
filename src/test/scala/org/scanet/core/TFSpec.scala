package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.using
import org.scanet.core.Tensor.scalar
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._

class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function composition" should "work" in {
    val sqr: TF1[Int, Int] = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg * arg
      (arg, result)
    })
    val identity: TF1[Int, Int] = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg
      (arg, result)
    })
    val leftPlusRightSqr = identity.compose(sqr)(_ + _)
    using(session => {
      val func = leftPlusRightSqr.compile(session)
      // we can call the function multiple times now
      // that is compiled and same session will be reused
      func(scalar(5), scalar(3)) should be(scalar(14))
      func(scalar(2), scalar(6)) should be(scalar(38))
    })
  }

}
