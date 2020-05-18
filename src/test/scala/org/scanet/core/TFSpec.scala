package org.scanet.core

import java.util.concurrent.atomic.AtomicInteger

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.using
import org.scanet.core.Tensor.scalar
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._

class TFSpec extends AnyFlatSpec with CustomMatchers {

  "memoization" should "cache result of the function and reuse it" in {
    val counter = new AtomicInteger(0)
    val plus = memoize((a: Int, b: Int) => {
      counter.incrementAndGet()
      a + b
    })
    counter.get() should be(0)
    plus(2, 3) should be(5)
    counter.get() should be(1)
    plus(2, 3) should be(5)
    counter.get() should be(1)
  }

  "tensor function composition" should "work" in {
    val sqr: TF1[Int, Int] = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg + arg
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
      func(scalar(5), scalar(3)) should be(scalar(11))
      // cool thing here is if we call the function with args which have
      // different shape - the new graph will be compiled and also cached
    })
  }

}
