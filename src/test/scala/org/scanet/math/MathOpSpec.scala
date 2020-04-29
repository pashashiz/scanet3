package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.{Shape, Tensor}
import org.scanet.math.syntax._

class MathOpSpec extends AnyFlatSpec with Matchers {

  "plus" should "add 2 scalars" in {
    (2.0f.const plus 5.0f.const).eval should be(Tensor.scalar(7.0f))
  }

  it should "add 2 tensors when one includes shape of the other" in {
    val a = Tensor.matrix(
      Array(1, 2),
      Array(1, 2))
    val b = Tensor.vector(1, 2)
    val c = Tensor.matrix(
      Array(2, 4),
      Array(2, 4))
    (a.const plus b.const).eval should be(c)
  }

  it should "work when adding same tensor" in {
    val a = 5.0f.const.as("a")
    (a plus a).as("c").eval should be(Tensor.scalar(10.0f))
  }

  it should "fail when 2 tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.matrix(Array(1, 2), Array(1, 2)).const plus Tensor.vector(1, 2, 3).const
    } should have message
      "requirement failed: cannot add tensors with shapes (2, 2) + (3)"
  }

  "multiply" should "produce dot product on 2 matrices" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(1, 2, 3))
    val b = Tensor.matrix(
      Array(1, 2),
      Array(1, 2),
      Array(1, 2))
    val c = Tensor.matrix(
      Array(6, 12),
      Array(6, 12))
    (a.const * b.const).eval should be(c)
  }

  it should "produce dot product of vector and matrix" in {
    val a = Tensor.vector(1, 2, 3)
    val b = Tensor.matrix(
      Array(1, 2),
      Array(1, 2),
      Array(1, 2))
    (a.const * b.const).eval should be(Tensor.vector(6, 12))
  }

  it should "multiply 2 scalars" in {
    (2.const * 3.const).eval should be(Tensor.scalar(6))
  }

  it should "multiply scalar and vector" in {
    (2.const * Tensor.vector(1, 2, 3).const).eval should be(Tensor.vector(2, 4, 6))
  }

  it should "fail to multiply vector and scalar" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const * 2.const).eval
    } should have message "requirement failed: cannot multiply tensors with shapes (1, 3) * (1, 1)"
  }

  it should "fail to multiply 3D tensors" in {
    the [IllegalArgumentException] thrownBy {
      val tensor = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2))
      (tensor.const * tensor.const).eval
    } should have message "requirement failed: rank cannot be > 2 but got tensors with shapes (2, 2, 2) * (2, 2, 2)"
  }

  "minus" should "subtract 2 scalars" in {
    (5.const - 3.const).eval should be(Tensor.scalar(2))
  }

  it should "subtract 2 tensors with same shape" in {
    (Tensor.vector(5, 10).const - Tensor.vector(1, 2).const).eval should be(Tensor.vector(4, 8))
  }

  it should "subtract 2 tensors when left includes shape of right" in {
    (Tensor.vector(5, 10).const - 2.const).eval should be(Tensor.vector(3, 8))
  }

  it should "subtract 2 tensors when right includes shape of left" in {
    (2.const - Tensor.vector(5, 10).const).eval should be(Tensor.vector(-3, -8))
  }

  it should "fail to subtract when 2 tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const - Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot subtracted tensors with shapes (2, 2) - (3)"
  }

  "negate" should "work on a scalar" in {
    3.const.negate.eval should be(Tensor.scalar(-3))
  }

  "negate" should "work on a tensor" in {
    Tensor.vector(1, 2, 3).const.negate.eval should be(Tensor.vector(-1, -2, -3))
  }

  "element-wise division" should "work on tensors with same dimensions" in {
    (Tensor.vector(5, 10, 15).const / Tensor.vector(5, 5, 5).const).eval should be(Tensor.vector(1, 2, 3))
  }

  it should "support broadcasting" in {
    (Tensor.vector(5, 10, 15).const / 5.const).eval should be(Tensor.vector(1, 2, 3))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const / Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot divide tensors with shapes (2, 2) / (3)"
  }

  "element-wise multiplication" should "work on tensors with same dimensions" in {
    (Tensor.vector(1, 2, 3).const :* Tensor.vector(5, 5, 5).const).eval should be(Tensor.vector(5, 10, 15))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const :* 5.const).eval should be(Tensor.vector(5, 10, 15))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const :* Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot multiply tensors with shapes (2, 2) :* (3)"
  }

  "element-wise equality" should "work on tensors with same dimensions" in {
    (Tensor.vector(1, 2, 3).const :== Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(true, true, false))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const :== 1.const).eval should be(Tensor.vector(true, false, false))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const :== Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot check for equality tensors with shapes (2, 2) :== (3)"
  }
}