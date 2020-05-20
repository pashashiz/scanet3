package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.using
import org.scanet.core.Tensor.scalar
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function of 1 arg" should "support complex result type" in {
    val duplicate = TF1(shape => {
      val arg = placeholder[Int](shape)
      // aka LISP
      (arg, (arg + 0.const, arg + 0.const))
    }).returns[(Tensor[Int], Tensor[Int])]
    using(session => {
      val func = duplicate.compile(session)
      // we can call the function multiple times now
      // that is compiled and same session will be reused
      // cool thing here is if we call the function with args which have
      // different shape - the new graph will be compiled and also cached
      func(scalar(5)) should be((scalar(5), scalar(5)))
    })
  }

  it should "support composition with other 1 arg function" in {
    val sqr = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg + arg
      (arg, result)
    }).returns[Tensor[Int]]
    val identity = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg
      (arg, result)
    }).returns[Tensor[Int]]
    val leftPlusRightSqr: TF2[Int, Int, Output[Int], Tensor[Int]] = identity.compose(sqr)(_ + _)
    using(session => {
      val func = leftPlusRightSqr.compile(session)
      func(scalar(5), scalar(3)) should be(scalar(11))
    })
  }

  it should "support composition with other 2 arg function" in {
    val identity = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg
      (arg, result)
    }).returns[Tensor[Int]]
    val plus = TF2((shape1, shape2) => {
      val arg1 = placeholder[Int](shape1)
      val arg2 = placeholder[Int](shape2)
      (arg1, arg2, arg1 + arg2)
    }).returns[Tensor[Int]]
    val leftMultiplyRightSum: TF3[Int, Int, Int, Output[Int], Tensor[Int]] = identity.compose(plus)(_ * _)
    using(session => {
      val func = leftMultiplyRightSum.compile(session)
      func(scalar(4), scalar(2), scalar(3)) should be(scalar(20))
    })
  }

  "tensor function of 2 args" should "work" in {
    val plus = TF2((shapeFirst, shapeSecond) => {
      val first = placeholder[Int](shapeFirst)
      val second = placeholder[Int](shapeSecond)
      (first, second, first + second)
    }).returns[Tensor[Int]]
    using(session => {
      val func = plus.compile(session)
      func(scalar(2), scalar(3)) should be(scalar(5))
    })
  }

  it should "support composition with other 1 arg function" in {
    val identity = TF1(shape => {
      val arg = placeholder[Int](shape)
      val result = arg
      (arg, result)
    }).returns[Tensor[Int]]
    val plus = TF2((shape1, shape2) => {
      val arg1 = placeholder[Int](shape1)
      val arg2 = placeholder[Int](shape2)
      (arg1, arg2, arg1 + arg2)
    }).returns[Tensor[Int]]
    val leftMultiplyRightSum: TF3[Int, Int, Int, Output[Int], Tensor[Int]] = identity.compose(plus)(_ * _)
    using(session => {
      val func = leftMultiplyRightSum.compile(session)
      func(scalar(4), scalar(2), scalar(3)) should be(scalar(20))
    })
  }

  "tensor function of 3 args" should "work" in {
    val plus = TF3((shapeFirst, shapeSecond, shapeThird) => {
      val first = placeholder[Int](shapeFirst)
      val second = placeholder[Int](shapeSecond)
      val third = placeholder[Int](shapeThird)
      (first, second, third, first + second + third)
    }).returns[Tensor[Int]]
    using(session => {
      val func = plus.compile(session)
      func(scalar(2), scalar(3), scalar(4)) should be(scalar(9))
    })
  }
}
