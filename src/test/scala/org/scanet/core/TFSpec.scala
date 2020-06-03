package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.withing
import org.scanet.core.Tensor.scalar
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function of 1 arg" should "support complex result type" in {
    val duplicate = TF1((arg: Output[Int]) =>
      (arg + 0.const, arg + 0.const)).returns[(Tensor[Int], Tensor[Int])]
    withing(session => {
      val func = duplicate.compile(session)
      // we can call the function multiple times now
      // that is compiled and same session will be reused
      // cool thing here is if we call the function with args which have
      // different shape - the new graph will be compiled and also cached
      func(scalar(5)) should be((scalar(5), scalar(5)))
    })
  }

  it should "support composition with other 1 arg function" in {
    val sqr = TF1((arg: Output[Int]) => arg + arg).returns[Tensor[Int]]
    val identity = TF1((arg: Output[Int]) => arg).returns[Tensor[Int]]
    val leftPlusRightSqr: TF2[Int, Int, Output[Int], Tensor[Int]] = identity.compose(sqr)(_ + _)
    withing(session => {
      val func = leftPlusRightSqr.compile(session)
      func(scalar(5), scalar(3)) should be(scalar(11))
    })
  }

  it should "support composition with other 2 arg function" in {
    val identity = TF1((arg: Output[Int]) => arg).returns[Tensor[Int]]
    val plus = TF2((arg1: Output[Int], arg2: Output[Int]) =>
      arg1 + arg2).returns[Tensor[Int]]
    val leftMultiplyRightSum = identity.compose(plus)(_ * _).into[Tensor[Int]]
    withing(session => {
      val func = leftMultiplyRightSum.compile(session)
      func(scalar(4), scalar(2), scalar(3)) should be(scalar(20))
    })
  }

  "tensor function of 2 args" should "work" in {
    val plus = TF2((arg1: Output[Int], arg2: Output[Int]) =>
      arg1 + arg2).returns[Tensor[Int]]
    withing(session => {
      val func = plus.compile(session)
      func(scalar(2), scalar(3)) should be(scalar(5))
    })
  }

  it should "support composition with other 1 arg function" in {
    val identity = TF1((arg: Output[Int]) => arg).returns[Tensor[Int]]
    val plus = TF2((arg1: Output[Int], arg2: Output[Int]) =>
      arg1 + arg2).returns[Tensor[Int]]
    val leftMultiplyRightSum = identity.compose(plus)(_ * _).into[Tensor[Int]]
    withing(session => {
      val func = leftMultiplyRightSum.compile(session)
      func(scalar(4), scalar(2), scalar(3)) should be(scalar(20))
    })
  }

  "tensor function of 3 args" should "work" in {
    val plus = TF3((arg1: Output[Int], arg2: Output[Int], arg3: Output[Int]) =>
      arg1 + arg2 + arg3).returns[Tensor[Int]]
    withing(session => {
      val func = plus.compile(session)
      func(scalar(2), scalar(3), scalar(4)) should be(scalar(9))
    })
  }
}
