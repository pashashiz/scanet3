package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.withing
import org.scanet.core.Tensor.scalar
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

/*_*/
class TFuSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function of 1 arg" should "work with plain outputs as arg" in {
    val inc = TFu1((arg: Output[Int]) => arg + 1.const)
    withing(session => {
      val func = inc compile session
      func(scalar(5)) should be(scalar(6))
    })
  }

  it should "work with seq of outputs as arg" in {
    val sum = TFu1((arg: OutputSeq[Int]) => plus(arg: _*))
    withing(session => {
      val func = sum compile session
      func(Seq(scalar(1), scalar(2))) should be(scalar(3))
    })
  }

  it should "return seq of outputs" in {
    // NOTE: sadly, we have to specify a result type so we
    // could have proper result type to compile it later :(
    val double: TFu1[Int, Tensor[Int], OutputSeq[Int]] =
      TFu1((arg: Output[Int]) => Seq(arg + 0.const, arg + 0.const))
    withing(session => {
      val func = double compile session
      func(scalar(5)) should be(Seq(scalar(5), scalar(5)))
    })
  }

  it should "return complex tuple" in {
    val complex: TFu1[Int, Tensor[Int], (Output[Int], OutputSeq[Int])] =
    TFu1((arg: Output[Int]) => (arg + 0.const, Seq(arg + 0.const, arg + 0.const)))
    withing(session => {
      val func = complex compile session
      func(scalar(5)) should be((scalar(5), Seq(scalar(5), scalar(5))))
    })
  }
}