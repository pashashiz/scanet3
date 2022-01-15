package scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Session.withing
import scanet.core.Tensor.scalar
import scanet.math.syntax._
import scanet.test.CustomMatchers
import scala.collection.immutable.Seq

/*_*/
class TFSpec extends AnyFlatSpec with CustomMatchers {

  "tensor function of 1 arg" should "work with plain outputs as arg" in {
    val inc = TF1((arg: Expr[Int]) => arg + 1.const)
    withing(session => {
      val func = inc compile session
      func(scalar(5)) should be(scalar(6))
    })
  }

  it should "work with seq of outputs as arg" in {
    val sum = TF1((arg: OutputSeq[Int]) => plus(arg))
    withing(session => {
      val func = sum compile session
      func(Seq(scalar(1), scalar(2))) should be(scalar(3))
    })
  }

  it should "return seq of outputs" in {
    // NOTE: sadly, we have to specify a result type so we
    // could have proper result type to compile it later :(
    val double: TF1[Int, Tensor[Int], OutputSeq[Int]] =
      TF1((arg: Expr[Int]) => Seq(arg + 0.const, arg + 0.const): OutputSeq[Int])
    withing(session => {
      val func = double compile session
      func(scalar(5)) should be(Seq(scalar(5), scalar(5)))
    })
  }

  it should "return complex tuple" in {
    val complex: TF1[Int, Tensor[Int], (Expr[Int], OutputSeq[Int])] =
      TF1((arg: Expr[Int]) => (arg + 0.const, Seq(arg + 0.const, arg + 0.const): OutputSeq[Int]))
    withing(session => {
      val func = complex compile session
      func(scalar(5)) should be((scalar(5), Seq(scalar(5), scalar(5))))
    })
  }

  it should "be combined with other 1 arg function" in {
    val inc1 = TF1((arg: Expr[Int]) => arg + 1.const)
    val inc2 = TF1((arg: Expr[Int]) => arg + 2.const)
    val diff = inc1.combine(inc2)(_ - _)
    withing(session => {
      val func = diff compile session
      func(scalar(2), scalar(2)) should be(scalar(-1))
    })
  }

  it should "have an identity" in {
    val identity1 = TF1.identity[Expr, Int]
    val identity2 = TF1.identity[Expr, Int]
    val sum = identity1.combine(identity2)(_ + _)
    withing(session => {
      val func = sum compile session
      func(scalar(2), scalar(2)) should be(scalar(4))
    })
  }

  it should "display a graph with given argument shape" ignore {
    val inc = TF1((arg: Expr[Int]) => arg + 1.const)
    inc.display(Seq(Shape()), label = "inc")
  }

  "tensor function of 2 args" should "work" in {
    val plus = TF2((left: Expr[Int], right: Expr[Int]) => left + right)
    withing(session => {
      val func = plus compile session
      func(scalar(1), scalar(2)) should be(scalar(3))
    })
  }

  it should "have an identity" in {
    val plus = TF3((a1: Expr[Int], a2: Expr[Int], a3: Expr[Int]) => a1 + a2 + a3)
    val identity = TF2.identity[Expr, Int, Expr, Int]
    val sum = plus.combine(identity) {
      case (p, (a1, a2)) => p + a1 + a2
    }
    withing(session => {
      val func = sum compile session
      func(scalar(1), scalar(2), scalar(3), scalar(4), scalar(5)) should be(scalar(15))
    })
  }

  it should "display a graph with given argument shapes" ignore {
    val plus = TF2((left: Expr[Int], right: Expr[Int]) => left + right)
    plus.display(Seq(Shape()), Seq(Shape()), label = "plus")
  }

  "tensor function of 3 args" should "work" in {
    val plus = TF3((a1: Expr[Int], a2: Expr[Int], a3: Expr[Int]) => a1 + a2 + a3)
    withing(session => {
      val func = plus compile session
      func(scalar(1), scalar(2), scalar(3)) should be(scalar(6))
    })
  }

  it should "be combined with other 2 arg function" in {
    val plus1 = TF3((a1: Expr[Int], a2: Expr[Int], a3: Expr[Int]) => a1 + a2 + a3)
    val plus2 = TF2((a1: Expr[Int], a2: Expr[Int]) => a1 + a2)
    val diff = plus1.combine(plus2)(_ - _)
    withing(session => {
      val func = diff compile session
      func(scalar(1), scalar(2), scalar(3), scalar(4), scalar(5)) should be(scalar(-3))
    })
  }

  "tensor function of 4 args" should "work" in {
    val plus =
      TF4((a1: Expr[Int], a2: Expr[Int], a3: Expr[Int], a4: Expr[Int]) => a1 + a2 + a3 + a4)
    withing(session => {
      val func = plus compile session
      func(scalar(1), scalar(2), scalar(3), scalar(4)) should be(scalar(10))
    })
  }

  "tensor function of 5 args" should "work" in {
    val plus =
      TF5((a1: Expr[Int], a2: Expr[Int], a3: Expr[Int], a4: Expr[Int], a5: Expr[Int]) =>
        a1 + a2 + a3 + a4 + a5)
    withing(session => {
      val func = plus compile session
      func(scalar(1), scalar(2), scalar(3), scalar(4), scalar(5)) should be(scalar(15))
    })
  }
}