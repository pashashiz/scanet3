package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.{Shape, Tensor}
import org.scanet.math.syntax._

class MathBaseOpSpec extends AnyFlatSpec with Matchers {

  "const" should "have zeros gradient if input is other const" in {
    val a = 2.const
    val b = 3.const
    (a grad b).eval should be(Tensor.scalar(0))
  }

  it should "have ones gradient if input is same const" in {
    val a = 2.const
    (a grad a).eval should be(Tensor.scalar(1))
  }

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

  it should "calculate zero gradient if both sides do not contain a differentiable variable" in {
    val a = 2.const
    val b = 3.const
    val x = 4.const
    ((a + b) grad x).eval should be(Tensor.scalar(0))
  }

  it should "calculate a gradient equals to 1 if left side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((x + a) grad x).eval should be(Tensor.scalar(1))
  }

  it should "calculate a gradient equals to 1 if right side is a differentiable variable" in {
    val a = 2.const
    val x = 3.const
    ((a + x) grad x).eval should be(Tensor.scalar(1))
  }

  it should "calculate a gradient equals to 2 if right and left side is a differentiable variable" in {
    val x = 2.const
    ((x + x) grad x).eval should be(Tensor.scalar(2))
  }

  "multiplication" should "produce dot product on 2 matrices" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6))
    val b = Tensor.matrix(
      Array(1, 2),
      Array(1, 2),
      Array(1, 2))
    val c = Tensor.matrix(
      Array(6, 12),
      Array(15, 30))
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

  it should "calculate zero gradient if both sides do not contain a differentiable variable" in {
    val a = 2.const
    val b = 3.const
    val x = 4.const
    ((a + b) grad x).eval should be(Tensor.scalar(0))
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

  it should "calculate zero gradient if both sides do not contain a differentiable variable" in {
    val a = 2.const
    val b = 3.const
    val x = 4.const
    ((a :* b) grad x).eval should be(Tensor.scalar(0))
  }

  it should "calculate a gradient equals to right side if left side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((x :* a) grad x).eval should be(Tensor.scalar(3))
  }

  it should "calculate a gradient equals to left side if right side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((a :* x) grad x).eval should be(Tensor.scalar(3))
  }

  it should "calculate a gradient equals to sum if right and left side is a differentiable variable" in {
    val x = Tensor.vector(3, 5).const
    ((x :* x) grad x).eval should be(Tensor.vector(6, 10))
  }

  "sum" should "calculate sum across all axises by default" in {
    Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum.eval should be(Tensor.scalar(21))
  }

  it should "support reducing along matrix columns" in {
    Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum(Seq(0)).eval should be(Tensor.vector(5, 7, 9))
  }

  it should "support reducing along matrix rows" in {
    Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum(Seq(1)).eval should be(Tensor.vector(6, 15))
  }

  it should "support reducing D4 tensors" in {
    println(Tensor(Array(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4), Shape(2, 2, 2, 2)).const.sum(Seq(0, 1)).eval)
  }
}