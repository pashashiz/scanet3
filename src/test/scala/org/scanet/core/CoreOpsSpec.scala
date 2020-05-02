package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.syntax._

class CoreOpsSpec extends AnyFlatSpec with Matchers {

  "const" should "be evaluated" in {
    5.0f.const.eval should be(Tensor.scalar(5.0f))
  }

  "evaluated tensor" should "be specialized" in {
    // todo: make const(5.0f)).eval specialized
    // this works: println(OpEval(const(5.0f)).eval.getClass)
    // this does not: println(const(5.0f).eval.getClass)
    // might need to specialize Op, but that is too much overhead, try to avoid that
    // however, we only really care about Tensor.toArray() to return primitive array which works anyway
    val tensor: Tensor[Float] = Session.run(5.0f.const)
    tensor.getClass.getName should endWith("Tensor$mcF$sp")
  }

  "product of 2 ops" should "be evaluated" in {
    (1.const, 2.const).eval should be((Tensor.scalar(1), Tensor.scalar(2)))
  }

  "product of 3 ops" should "be evaluated" in {
    (1.const, 2.const, 3.const).eval should be((Tensor.scalar(1), Tensor.scalar(2), Tensor.scalar(3)))
  }

  "vector" should "be reshaped into matrix" in {
    Tensor.vector(1, 2, 3, 4).const.reshape(2, 2).eval should be(Tensor.matrix(Array(1, 2), Array(3, 4)))
  }

  "matrix" should "be reshaped into vector" in {
    Tensor.matrix(Array(1, 2), Array(3, 4)).const.reshape(4).eval should be(Tensor.vector(1, 2, 3, 4))
  }

  "matrix" should "be reshaped into other matrix" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6))
    val b = Tensor.matrix(
      Array(1, 2),
      Array(3, 4),
      Array(5, 6))
    a.const.reshape(3, 2).eval should be(b)
  }

  "matrix" should "fail to reshape when power does not match" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.range(0 until 7).const.reshape(4, 4)
    } should have message "requirement failed: shape (7) cannot be reshaped into (4, 4)"
  }

  "matrix" should "be squeezed into vector when first dimension is 1" in {
    Tensor.matrix(Array(1, 2, 3)).const.squeeze.eval should be(Tensor.vector(1, 2, 3))
  }

  "vector of Floats" should "be casted into Ints" in {
    Tensor.vector(1.2f, 2.2f, 3.3f).const.cast[Int].eval should be(Tensor.vector(1, 2, 3))
  }

  "assert" should "verify current output" in {
    val a = Tensor.vector(1, 2).const
    val b = Tensor.vector(3, 4).const
    val c = Tensor.vector(4, 6).const

    (a plus b).assert(_ === c).eval should be(Tensor.vector(4, 6))
  }

  "assert" should "throw when current condition is false" in {
    the [IllegalArgumentException] thrownBy {
      val a = Tensor.vector(1, 2).const
      val b = Tensor.vector(3, 4).const
      val c = Tensor.vector(5, 6).const

      (a plus b).assert(_ === c).eval
    } should have message "assertion failed: [4 6]\n\t [[{{node Assert_0}}]]"
  }

  "assert that" should "verify dependent condition" in {
    val a = 1.const
    val b = 2.const
    val c = (a plus b) << assertThat((a gt 0.const) and (b gt 0.const))

    c.eval should be(Tensor.scalar(3))
  }

  "assert that" should "should fail when dependent condition is false" in {
    the [IllegalArgumentException] thrownBy {
      val a = Tensor.vector(1, 2).const
      val b = 0.const
      val c = (a div b) << assertThat(b gt 0.const, b)

      c.eval should be(Tensor.scalar(3))
    } should have message "assertion failed: [0]\n\t [[{{node Assert_0}}]]"
  }
}
