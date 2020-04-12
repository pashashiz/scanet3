package org.scanet.linalg

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.linalg.Op.{const, plus}
import org.scanet.syntax.core._

class OpSpec extends AnyFlatSpec with Matchers {

  "op example" should "work" in {
    val expr = const(5.0f, "a")
    val tensor: Tensor[Float] = Session.run(expr)
    println(tensor.show())
  }

  "plus" should "work" in {
    plus(const(Tensor.matrix(Array(1, 2, 3), Array(1, 2, 3)), "a"), const(Tensor.vector(1, 2), "b")).eval should be(Tensor.vector(1, 2, 3))
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
