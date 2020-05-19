package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.using
import org.scanet.core.Tensor.scalar
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function composition" should "work" in {
    val sqr: TF1[Int, Output[Int], Tensor[Int]] = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg + arg
      (arg, result)
    })
    val identity: TF1[Int, Output[Int], Tensor[Int]] = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg
      (arg, result)
    })
    val leftPlusRightSqr: TF2[Int, Int, Output[Int], Tensor[Int]] = identity.compose(sqr)(_ + _)
    using(session => {
      val func: (Tensor[Int], Tensor[Int]) => Tensor[Int] = leftPlusRightSqr.compile(session)
      // we can call the function multiple times now
      // that is compiled and same session will be reused
      func(scalar(5), scalar(3)) should be(scalar(11))
      // cool thing here is if we call the function with args which have
      // different shape - the new graph will be compiled and also cached
    })
  }

}
