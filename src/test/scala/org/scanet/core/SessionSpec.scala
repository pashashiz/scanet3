package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.using
import org.scanet.syntax.placeholder
import org.scanet.test.CustomMatchers

import org.scanet.math.syntax._

class SessionSpec extends AnyFlatSpec with CustomMatchers {

  "placeholder" should "be substituted with a session" in {
    using(session => {
      val a = placeholder[Int]()
      val b = 10.const
      val c = a + b
      session.runner
        .feed(a -> Tensor.scalar(5))
        .eval(c) should be(Tensor.scalar(15))
    })
  }

  "session" should "eval generically when single output is used" in {
    using(session => {
      session.runner.evalX[Output[Int], Tensor[Int]](10.const) should be(Tensor.scalar(10))
    })
  }

  it should "eval generically when tuple 2 is used" in {
    using(session => {
      session.runner.evalX[(Output[Int], Output[Int]), (Tensor[Int], Tensor[Int])](
        (10.const, 5.const)) should be((Tensor.scalar(10), Tensor.scalar(5)))
    })
  }

  it should "eval generically when tuple 3 is used" in {
    using(session => {
      session.runner.evalX[(Output[Int], Output[Int], Output[Int]), (Tensor[Int], Tensor[Int], Tensor[Int])](
        (10.const, 5.const, 1.const)) should be((Tensor.scalar(10), Tensor.scalar(5), Tensor.scalar(1)))
    })
  }
}
