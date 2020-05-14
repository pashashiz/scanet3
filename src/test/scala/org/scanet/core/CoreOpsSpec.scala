package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Session.using
import org.scanet.core.Slice.syntax.::
import org.scanet.syntax._

class CoreOpsSpec extends AnyFlatSpec with Matchers {

  "const" should "be evaluated" in {
    5.0f.const.eval should be(Tensor.scalar(5.0f))
  }

  "product of 2 ops" should "be evaluated" in {
    (1.const, 2.const).eval should be((Tensor.scalar(1), Tensor.scalar(2)))
  }

  "product of 3 ops" should "be evaluated" in {
    (1.const, 2.const, 3.const).eval should be((Tensor.scalar(1), Tensor.scalar(2), Tensor.scalar(3)))
  }

  "reshape" should "transform vector into matrix" in {
    Tensor.vector(1, 2, 3, 4).const.reshape(2, 2).eval should be(Tensor.matrix(Array(1, 2), Array(3, 4)))
  }

  it should "transform matrix into vector" in {
    Tensor.matrix(Array(1, 2), Array(3, 4)).const.reshape(4).eval should be(Tensor.vector(1, 2, 3, 4))
  }

  it should "transform matrix into another matrix" in {
    Tensor.matrix(Array(1, 2), Array(3, 4)).const.reshape(4).eval should be(Tensor.vector(1, 2, 3, 4))
  }

  it should "fail when power does not match" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.range(0 until 7).const.reshape(4, 4)
    } should have message "requirement failed: shape (7) cannot be reshaped into (4, 4)"
  }

  it should "support gradient" in {
    val x = Tensor.vector(1, 2, 3, 4).const
    x.reshape(2, 2).sum.grad(x).eval should be(Tensor.vector(1, 1, 1, 1))
  }

  "squeeze" should "convert matrix into a vector when first dimension is 1" in {
    Tensor.matrix(Array(1, 2, 3)).const.squeeze.eval should be(Tensor.vector(1, 2, 3))
  }

  it should "support gradient" in {
    val x = Tensor.matrix(Array(1, 2, 3)).const
    x.squeeze.sum.grad(x).eval should be(Tensor.matrix(Array(1, 1, 1)))
  }

  "cast" should "convert vector of Floats into Ints" in {
    Tensor.vector(1.2f, 2.2f, 3.3f).const.cast[Int].eval should be(Tensor.vector(1, 2, 3))
  }

  "cast" should "support gradient" in {
    val a = Tensor.vector(1, 2, 3).const
    val x = Tensor.vector(5.2f, 10.2f, 15.3f).const
    ((x.cast[Int] + a).sum grad x).eval should be(Tensor.vector(1.0f, 1.0f, 1.0f))
  }

  "when" should "calculate output based on true condition" in {
    val a = 1.const
    val b = 0.const
    val c = 2.const

    val ternary = when(a gt b) thenDo (a plus c) elseDo (a minus c)
    ternary.eval should be(Tensor.scalar(3))
  }

  it should "calculate output based on false condition" in {
    val a = (-1).const
    val b = 0.const
    val c = 2.const

    val ternary = when(a gt b) thenDo (a plus c) elseDo (a minus c)
    ternary.eval should be(Tensor.scalar(-3))
  }

  "placeholder" should "be substituted with a session" in {
    using(session => {
      val a = placeholder[Int]()
      val b = 10.const
      val c = a + b
      session.runner
        .feed(a -> Tensor.scalar(5))
        .eval(c) should be(Tensor.scalar(15))
    })
  }

  "slice" should "index a matrix to a row" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.slice(0).eval should be(Tensor.vector(1, 2, 3))
  }

  it should "index a matrix to an element" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.slice(0, 0).eval should be(Tensor.scalar(1))
  }

  it should "slice a matrix" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.slice(::, 0 until 2).eval should be(Tensor.matrix(Array(1, 2), Array(4, 5)))
  }

  it should "fail when out of bounds" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.eye[Int](3).const.slice(1, 1, 1)
    } should have message "requirement failed: projection (1, 1, 1) is out of bound, should fit shape (3, 3)"
  }
}
