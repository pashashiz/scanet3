package org.scanet.linalg

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.linalg.Op.{const, plus}
import org.scanet.syntax.core._
import org.scanet.linalg.Eval.syntax._

class OpSpec extends AnyFlatSpec with Matchers {

  "op example" should "work" in {
    val expr = const(5.0f, "a")
    val tensor: Tensor[Float] = Session.run(expr)
    println(tensor.show())
  }

  "plus" should "work" in {
    println(plus(
      const(Tensor.matrix(Array(1, 2), Array(1, 2))),
      const(Tensor.vector(1, 2))))
    plus(
      const(Tensor.matrix(Array(1, 2), Array(1, 2))),
      const(Tensor.vector(1, 2)))
      .eval should be(Tensor.matrix(Array(2, 4), Array(2, 4)))
  }

  "product of 2 ops" should "be evaluated" in {
    (const(1), const(2)).eval should be((Tensor.scalar(1), Tensor.scalar(2)))
  }

  "plus same value" should "work" in {
    val a = const(5.0f, "a")
    plus(a, a, "c").eval should be(Tensor.scalar(10.0f))
  }

  "resulting tensor" should "be specialized" in {
    val tensor: Tensor[Float] = Session.run(const(5.0f, "a"))
    tensor.getClass.getName should endWith("Tensor$mcF$sp")
  }
}
