package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
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

  "join" should "concat 2 vectors" in {
    val a = Tensor.vector(1, 2).const
    val b = Tensor.vector(3, 4).const
    (a join b).eval should be(Tensor.vector(1, 2, 3, 4))
  }

  it should "concat 2 matrices rows" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6)).const
    val b = Tensor.matrix(
      Array(7, 8, 9),
      Array(10, 11, 12)).const
    val ab = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6),
      Array(7, 8, 9),
      Array(10, 11, 12))
    (a join b).eval should be(ab)
  }

  it should "concat 2 matrices columns" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6)).const
    val b = Tensor.matrix(
      Array(7, 8, 9),
      Array(10, 11, 12)).const
    val ab = Tensor.matrix(
      Array(1, 2, 3, 7, 8, 9),
      Array(4, 5, 6, 10, 11, 12))
    a.joinAlong(b, 1).eval should be(ab)
  }

  it should "fail when dimensions do not match" in {
    the [IllegalArgumentException] thrownBy {
      val a = Tensor.matrix(
        Array(1, 2, 3),
        Array(4, 5, 6)).const
      val b = Tensor.matrix(
        Array(7, 8),
        Array(9, 10)).const
      a join b
    } should have message "requirement failed: " +
      "all inputs should have same dimensions except the axis, but was (2, 3), (2, 2)"
  }

  "zip" should "pack 2 vectors into a matrix" in {
    val first = Tensor.vector(1, 2, 3).const
    val second = Tensor.vector(4, 5, 6).const
    val expected = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6))
    (first zip second).eval should be (expected)
  }

  "unzip" should "unpack 2 vectors from a matrix" in {
    val matrix = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6)).const
    matrix.unzip.eval should be((Tensor.vector(1, 2, 3), Tensor.vector(4, 5, 6)))
  }
}
