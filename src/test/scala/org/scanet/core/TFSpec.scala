package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._
import org.scanet.core.Session.withing
import org.scanet.core.Tensor.scalar

/*_*/
class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function of 1 arg" should "support complex result type" in {
    val duplicate = TF1((arg: Id[Output[Int]]) =>
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
  }

  it should "support sequence arg type" in {
    val f = TF1((arg: Seq[Output[Int]]) => plus(arg: _*).toId).returns[Id[Tensor[Int]]]
    val total = f.compile()
    total(Seq(scalar(1), scalar(3), scalar(5))) should be(scalar(9))
  }

  it should "support sequence result type" in {
    val f = TF1((arg: Id[Output[Int]]) =>
      Seq(fromId(arg) + 0.const, fromId(arg) + 0.const))
      .returns[Seq[Tensor[Int]]]
    val double = f.compile()
    double(scalar(1)) should be(Seq(scalar(1), scalar(1)))
  }

  it should "support sequence result type inside a tuple" in {
    val f = TF1((arg: Id[Output[Int]]) => {
      val a: Output[Int] = arg
      (1f.const.toId, Seq(a + 0.const, a + 0.const))
    }).returns[(Id[Tensor[Float]], Seq[Tensor[Int]])]
    val double = f.compile()
    double(scalar(1)) should be((scalar(1f), Seq(scalar(1), scalar(1))))
  }

  "tensor function of 2 args" should "work" in {
    val plus = TF2((arg1: Id[Output[Int]], arg2: Id[Output[Int]]) =>
      (fromId(arg1) + arg2).toId).returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3)) should be(scalar(5))
    })
  }

  "tensor function of 3 args" should "work" in {
    val plus = TF3((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3).toId).returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3), scalar(4)) should be(scalar(9))
    })
  }

  it should "support composition with other 2 arg function" in {
    val plus = TF3((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3).toId)
      .returns[Id[Tensor[Int]]]
    val minus = TF2((arg1: Id[Output[Int]], arg2: Id[Output[Int]]) =>
      (fromId(arg1) - arg2).toId)
      .returns[Id[Tensor[Int]]]
    val leftMultiplyRightSum = plus.compose(minus)(
      (left: Id[Output[Int]], right: Id[Output[Int]]) => (fromId(left) * right).toId).into[Id[Tensor[Int]]]
    withing(session => {
      val func = leftMultiplyRightSum compile session
      func(scalar(1), scalar(2), scalar(3), scalar(4), scalar(2)) should be(scalar(12))
    })
  }

  "tensor function of 4 args" should "work" in {
    val plus = TF4((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]], arg4: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3 + arg4).toId)
      .returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3), scalar(4), scalar(5)) should be(scalar(14))
    })
  }

  "tensor function of 5 args" should "work" in {
    val plus = TF5((arg1: Id[Output[Int]], arg2: Id[Output[Int]], arg3: Id[Output[Int]], arg4: Id[Output[Int]], arg5: Id[Output[Int]]) =>
      (fromId(arg1) + arg2 + arg3 + arg4 + arg5).toId)
      .returns[Id[Tensor[Int]]]
    withing(session => {
      val func = plus compile session
      func(scalar(2), scalar(3), scalar(4), scalar(5), scalar(6)) should be(scalar(20))
    })
  }
}
/*_*/
