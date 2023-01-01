package scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Session.withing
import scanet.core.Tensor.scalar
import scanet.math.syntax._
import scanet.test.CustomMatchers
import scala.collection.immutable.Seq

class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function of 1 arg" should "work with plain outputs as arg" in {
    val inc = (arg: Expr[Int]) => arg + 1.const
    val func = inc.compile
    func(scalar(5)) should be(scalar(6))
  }

  it should "work with custom session" in {
    val inc = (arg: Expr[Int]) => arg + 1.const
    withing { session =>
      val func = inc compileWith session
      func(scalar(5)) should be(scalar(6))
    }
  }

  it should "work with seq of outputs as arg" in {
    val sum = (arg: Seq[Expr[Int]]) => plus(arg)
    val func = sum.compile
    func(Seq(scalar(1), scalar(2))) should be(scalar(3))
  }

  it should "return seq of outputs" in {
    val double = (arg: Expr[Int]) => Seq(arg + 0.const, arg + 1.const)
    val func = double.compile
    func(scalar(5)) should be(Seq(scalar(5), scalar(6)))
  }

  it should "return complex tuple" in {
    val complex = (arg: Expr[Int]) => (arg + 0.const, Seq(arg + 1.const, arg + 2.const))
    val func = complex.compile
    func(scalar(5)) should be((scalar(5), Seq(scalar(6), scalar(7))))
  }

  "tensor function of 2 args" should "work" in {
    val plus = (left: Expr[Int], right: Expr[Int]) => left + right
    val func = plus.compile
    func(scalar(1), scalar(2)) should be(scalar(3))
  }

  "tensor function of 3 args" should "work" in {
    val plus = (a1: Expr[Int], a2: Expr[Int], a3: Expr[Int]) => a1 + a2 + a3
    val func = plus.compile
    func(scalar(1), scalar(2), scalar(3)) should be(scalar(6))
  }

  "tensor function of 4 args" should "work" in {
    val plus = (a1: Expr[Int], a2: Expr[Int], a3: Expr[Int], a4: Expr[Int]) => a1 + a2 + a3 + a4
    val func = plus.compile
    func(scalar(1), scalar(2), scalar(3), scalar(4)) should be(scalar(10))
  }

  "tensor function of 5 args" should "work" in {
    val plus = (a1: Expr[Int], a2: Expr[Int], a3: Expr[Int], a4: Expr[Int], a5: Expr[Int]) =>
      a1 + a2 + a3 + a4 + a5
    val func = plus.compile
    func(scalar(1), scalar(2), scalar(3), scalar(4), scalar(5)) should be(scalar(15))
  }
}
