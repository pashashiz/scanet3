package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._
import org.scanet.core.Session.syntaxX._
import org.scanet.core.Session.withing
import org.scanet.core.Tensor.scalar

class TFxSpec extends AnyFlatSpec with CustomMatchers {

  // todo: test on Seq input and output

  "tensor function of 1 arg" should "support complex result type" in {
    /*_*/
    val duplicate = TFx1((arg: Id[Output[Int]]) =>
      ((0.const + arg).toId, (0.const + arg).toId))
      .returns[(Id[Tensor[Int]], Id[Tensor[Int]])]
    withing(session => {
      val func = duplicate compile session
      // we can call the function multiple times now
      // that is compiled and same session will be reused
      // cool thing here is if we call the function with args which have
      // different shape - the new graph will be compiled and also cached
      func(scalar(5)) should be((scalar(5), scalar(5)))
    })
    /*_*/
  }

  "tensor function of 2 args" should "work" in {
    /*_*/
    val plus = TFx2((arg1: Id[Output[Int]], arg2: Id[Output[Int]]) =>
      (fromId(arg1) + arg2).toId).returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3)) should be(scalar(5))
    })
    /*_*/
  }

  "tensor function of 3 args" should "work" in {
    /*_*/
    val plus = TFx3((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3).toId).returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3), scalar(4)) should be(scalar(9))
    })
    /*_*/
  }

  it should "support composition with other 2 arg function" in {
    /*_*/
    val plus = TFx3((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3).toId)
      .returns[Id[Tensor[Int]]]
    val minus = TFx2((arg1: Id[Output[Int]], arg2: Id[Output[Int]]) =>
      (fromId(arg1) - arg2).toId)
      .returns[Id[Tensor[Int]]]
    val leftMultiplyRightSum = plus.compose(minus)(
      (left: Id[Output[Int]], right: Id[Output[Int]]) => (fromId(left) * right).toId).into[Id[Tensor[Int]]]
    withing(session => {
      val func = leftMultiplyRightSum compile session
      func(scalar(1), scalar(2), scalar(3), scalar(4), scalar(2)) should be(scalar(12))
    })
    /*_*/
  }

  "tensor function of 4 args" should "work" in {
    /*_*/
    val plus = TFx4((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]], arg4: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3 + arg4).toId)
      .returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3), scalar(4), scalar(5)) should be(scalar(14))
    })
    /*_*/
  }

  "tensor function of 5 args" should "work" in {
    /*_*/
    val plus = TFx5((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]], arg4: Id[Output[Int]], arg5: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3 + arg4 + arg5).toId)
      .returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3), scalar(4), scalar(5), scalar(6)) should be(scalar(20))
    })
    /*_*/
  }
}
