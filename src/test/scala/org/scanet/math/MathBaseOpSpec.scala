package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.{Output, Shape, Tensor}
import org.scanet.math.syntax._

import scala.Array.range

class MathBaseOpSpec extends AnyFlatSpec with Matchers {

  "const" should "have ones gradient if input is same const" in {
    val a = 2.const
    (a grad a).returns[Float].eval should be(Tensor.scalar(1))
  }

  it should "fail to find a gradient if input is not a part of the computation graph" in {
    the [IllegalArgumentException] thrownBy {
      val a = 2.const
      val b = 3.const
      (a grad b).returns[Float]
    } should have message "requirement failed: " +
      "cannot find a gradient with respect to Const[Int]:() cause that input is not a part of the computation graph"
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

  it should "calculate a gradient if left side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((x + a) grad x).returns[Float].eval should be(Tensor.scalar(1))
  }

  it should "calculate a gradient if right side is a differentiable variable" in {
    val a = 2.const
    val x = 3.const
    ((a + x) grad x).returns[Float].eval should be(Tensor.scalar(1))
  }

  it should "calculate a gradient if right and left side is a differentiable variable" in {
    val x = 2.const
    ((x + x) grad x).returns[Float].eval should be(Tensor.scalar(2))
  }

  it should "calculate a gradient for minus op if left side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((x - a) grad x).returns[Float].eval should be(Tensor.scalar(1))
  }

  it should "calculate a gradient for minus op if right side is a differentiable variable" in {
    val a = 2.const
    val x = 3.const
    ((a - x) grad x).returns[Float].eval should be(Tensor.scalar(-1))
  }

  it should "calculate a gradient for minus op if right and left side is a differentiable variable" in {
    val x = 2.const
    ((x - x) grad x).returns[Float].eval should be(Tensor.scalar(0))
  }

  it should "calculate a gradient with broadcasting when smaller tensor is a differentiable variable" in {
    val a = Tensor.matrix(
      Array(5, 10, 15),
      Array(20, 25, 30)).const
    val x = Tensor.vector(1, 2, 3).const
    val grad = Tensor.vector(2, 2, 2)
    ((a + x).sum grad x).returns[Float].eval should be(grad)
    ((x + a).sum grad x).returns[Float].eval should be(grad)
  }

  it should "calculate a gradient with broadcasting when bigger tensor is a differentiable variable" in {
    val x = Tensor.matrix(
      Array(5, 10, 15),
      Array(20, 25, 30)).const
    val a = Tensor.vector(1, 2, 3).const
    val grad = Tensor.matrix(
      Array(1, 1, 1),
      Array(1, 1, 1))
    ((x + a).sum grad x).returns[Float].eval should be(grad)
    ((a + x).sum grad x).returns[Float].eval should be(grad)
  }

  "plus N" should "add multiple tensors" in {
    plus(1.const, 2.const, 3.const).eval should be(Tensor.scalar(6))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      plus(1.const, 2.const, Tensor.vector(3, 4).const).eval
    } should have message
      "requirement failed: shapes of all tensors should be the same, but was () + () + (2)"
  }

  it should "calculate gradient for plus N" in {
    val a = 1.const
    val b = 2.const
    val x = 3.const

    plus(a, b, x).sum.grad(x).returns[Float].eval should be(Tensor.scalar(1))
  }

  it should "calculate gradient for plus N when diff variable is used more than once" in {
    val a = Tensor.vector(1, 2).const
    val x = Tensor.vector(5, 6).const

    plus(a, x, x).sum.grad(x).returns[Float].eval should be(Tensor.vector(2, 2))
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

  it should "calculate gradient when 2 matrices are given and right side is a differentiable variable" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6)).const
    val x = Tensor.matrix(
      Array(5, 10),
      Array(15, 20),
      Array(25, 30)).const
    val grad = Tensor.matrix(
      Array(5, 5),
      Array(7, 7),
      Array(9, 9))
    ((a * x).sum grad x).returns[Float].eval should be(grad)
  }

  it should "calculate gradient when 2 matrices are given and left side is a differentiable variable" in {
    val x = Tensor.matrix(
      Array(1, 2, 3),
      Array(4, 5, 6))
      .const
    val a = Tensor.matrix(
      Array(5, 10),
      Array(15, 20),
      Array(25, 30))
      .const
    val grad = Tensor.matrix(
      Array(15, 35, 55),
      Array(15, 35, 55))
    ((x * a).sum grad x).returns[Float].eval should be(grad)
  }

  it should "calculate gradient when vector and matrix are given and left side is a differentiable variable" in {
    val x = Tensor.vector(1, 2, 3).const
    val a = Tensor.matrix(
      Array(5, 10),
      Array(15, 20),
      Array(25, 30)).const
    (x * a).sum.grad(x).returns[Float].eval should be(Tensor.vector(15, 35, 55))
  }

  it should "calculate gradient when vector and matrix are given and right side is a differentiable variable" in {
    val a = Tensor.vector(1, 2, 3).const
    val x = Tensor.matrix(
      Array(5, 10),
      Array(15, 20),
      Array(25, 30)).const
    (a * x).sum.grad(x).returns[Float].eval should be(Tensor.matrix(
      Array(1, 1),
      Array(2, 2),
      Array(3, 3)))
  }

  it should "calculate gradient when scalar and vector are given and left side is a differentiable variable" in {
    val x = 2.const
    val a = Tensor.vector(1, 2, 3).const
    (x * a).sum.grad(x).returns[Float].eval should be(Tensor.scalar(6))
  }

  it should "calculate gradient when scalar and vector are given and right side is a differentiable variable" in {
    val a = 2.const
    val x = Tensor.vector(1, 2, 3).const
    (a * x).sum.grad(x).returns[Float].eval should be(Tensor.vector(2, 2, 2))
  }

  it should "calculate gradient when 2 scalars are given and left side is a differentiable variable" in {
    val x = 2.const
    val a = 3.const
    (x * a).grad(x).returns[Float].eval should be(Tensor.scalar(3))
  }

  it should "calculate gradient when 2 scalars are given and right side is a differentiable variable" in {
    val a = 2.const
    val x = 3.const
    (a * x).grad(x).returns[Float].eval should be(Tensor.scalar(2))
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

  it should "calculate a gradient with broadcasting when smaller tensor is a differentiable variable" in {
    val a = Tensor.matrix(
      Array(5, 10, 15),
      Array(20, 25, 30)).const
    val x = Tensor.vector(1, 2, 3).const
    ((a - x).sum grad x).returns[Float].eval should be(Tensor.vector(-2, -2, -2))
    ((x - a).sum grad x).returns[Float].eval should be(Tensor.vector(2, 2, 2))
  }

  it should "calculate a gradient with broadcasting when bigger tensor is a differentiable variable" in {
    val x = Tensor.matrix(
      Array(5, 10, 15),
      Array(20, 25, 30)).const
    val a = Tensor.vector(1, 2, 3).const
    ((x - a).sum grad x).returns[Float].eval should be(Tensor.matrix(Array(1, 1, 1), Array(1, 1, 1)))
    ((a - x).sum grad x).returns[Float].eval should be(Tensor.matrix(Array(-1, -1, -1), Array(-1, -1, -1)))
  }

  "negate" should "work on a scalar" in {
    3.const.negate.eval should be(Tensor.scalar(-3))
  }

  it should "support gradient negation for scalars" in {
    val x: Output[Int] = 3.const
    x.negate.grad(x).returns[Float].eval should be(Tensor.scalar(-1))
  }

  "negate" should "work on a tensor" in {
    Tensor.vector(1, 2, 3).const.negate.eval should be(Tensor.vector(-1, -2, -3))
  }

  it should "support gradient of negation for vectors" in {
    val x = Tensor.vector(1, 2, 3).const
    x.negate.sum.grad(x).returns[Float].eval should be(Tensor.vector(-1, -1, -1))
  }

  "element-wise division" should "work on tensors with same dimensions" in {
    (Tensor.vector(5, 10, 15).const / Tensor.vector(5, 5, 5).const).eval should be(Tensor.vector(1, 2, 3))
  }

  it should "work for floating point numbers" in {
    (Tensor.vector(2.0f, 4.0f, 6.0f).const / Tensor.vector(10.0f, 10.0f, 10.0f).const).eval should be(Tensor.vector(0.2f, 0.4f, 0.6f))
  }

  it should "support broadcasting" in {
    (Tensor.vector(5, 10, 15).const / 5.const).eval should be(Tensor.vector(1, 2, 3))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const / Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot divide tensors with shapes (2, 2) / (3)"
  }

  it should "calculate gradient for constants" in {
    val a = 4.const
    val x = 2.const

    (a div x).sum.grad(x).returns[Float].eval should be(Tensor.scalar(-1.0))
    (a div x).sum.grad(a).returns[Float].eval should be(Tensor.scalar(0.5))
  }

  it should "calculate gradient for vectors" in {
    val a = Tensor.vector(5, 10, 15).const
    val x = Tensor.vector(5, 5, 5).const

    (a div x).sum.grad(x).returns[Float].eval should be(Tensor.vector(-0.2f, -0.4f, -0.6f))
    (a div x).sum.grad(a).returns[Float].eval should be(Tensor.vector(0.2f, 0.2f, 0.2f))
  }

  it should "calculate gradient for matrices with broadcasting when smaller tensor is differentiable value" in {
    val a = Tensor.matrix(
      Array(12, 16, 20),
      Array(24, 28, 32)).const
    val x = Tensor.vector(2, 4, 8).const

    (a div x).sum.grad(x).returns[Float].eval should be(Tensor.vector(-9f, -2.75f, -0.8125f))
    (x div a).sum.grad(x).returns[Float].eval should be(Tensor.vector(0.125f, 0.09821428f, 0.08125f))
  }

  it should "calculate gradient for matrices with broadcasting when bigger tensor is differentiable value" in {
    val a = Tensor.matrix(
      Array(12, 16, 20),
      Array(24, 28, 32)).const
    val x = Tensor.vector(2, 4, 8).const

    (a div x).sum.grad(a).returns[Float].eval should be(Tensor.matrix(Array(0.5f, 0.25f, 0.125f), Array(0.5f, 0.25f, 0.125f)))
    val grad = Tensor.matrix(
      Array(-0.013888889f, -0.015625f, -0.02f),
      Array(-0.0034722222f, -0.0051020407f, -0.0078125f))
    (x div a).sum.grad(a).returns[Float].eval should be(grad)
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

  it should "calculate a gradient equals to right side if left side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((x :* a) grad x).returns[Float].eval should be(Tensor.scalar(3))
  }

  it should "calculate a gradient equals to left side if right side is a differentiable variable" in {
    val a = 3.const
    val x = 2.const
    ((a :* x) grad x).returns[Float].eval should be(Tensor.scalar(3))
  }

  it should "calculate a gradient equals to sum if right and left side is a differentiable variable" in {
    val x = Tensor.scalar(3).const
    ((x :* x) grad x).returns[Float].eval should be(Tensor.scalar(6))
  }

  it should "calculate a gradient with broadcasting when smaller tensor is a differentiable variable" in {
    val a = Tensor.matrix(
      Array(5, 10, 15),
      Array(20, 25, 30)).const
    val x = Tensor.vector(1, 2, 3).const
    val grad = Tensor.vector(25, 35, 45)
    ((a :* x).sum grad x).returns[Float].eval should be(grad)
    ((x :* a).sum grad x).returns[Float].eval should be(grad)
  }

  it should "calculate a gradient with broadcasting when bigger tensor is a differentiable variable" in {
    val x = Tensor.matrix(
      Array(5, 10, 15),
      Array(20, 25, 30)).const
    val a = Tensor.vector(1, 2, 3).const
    val grad = Tensor.matrix(
      Array(1, 2, 3),
      Array(1, 2, 3))
    ((x :* a).sum grad x).returns[Float].eval should be(grad)
    ((a :* x).sum grad x).returns[Float].eval should be(grad)
  }

  "pow" should "compute the power of the tensor with high exponent" in {
    Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))
  }

  it should "return same tensor if exponent is 1" in {
    Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(1).eval should be(Tensor.vector(1.0f, 2.0f, 3.0f))
  }

  it should "return ones tensor if exponent is 0" in {
    Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(0).eval should be(Tensor.vector(1.0f, 1.0f, 1.0f))
  }

  it should "calculate a gradient" in {
    val x = Tensor.vector(5, 10, 15).const
    x.pow(3).sum.grad(x).returns[Float].eval should be(Tensor.vector(75.0f, 300.0f, 675.0f))
  }

  "sqrt" should "compute square root of tensor" in {
    Tensor.vector(1.0f, 4.0f, 9.0f).const.sqrt.eval should be(Tensor.vector(1.0f, 2.0f, 3.0f))
  }

  it should "calculate a gradient" in {
    val x = Tensor.vector(1.0f, 4.0f, 16.0f).const
    x.sqrt.sum.grad(x).returns[Float].eval should be(Tensor.vector(0.5f, 0.25, 0.125f))
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

  it should "support reducing 4D tensors" in {
    val tensor = Tensor(Array(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4), Shape(2, 2, 2, 2))
    tensor.const.sum(Seq(0, 1)).eval should be(Tensor.matrix(Array(1, 2), Array(3, 4)))
  }

  "transpose" should "be identity op on a scalar" in {
    Tensor.scalar(5).const.transpose.eval should be(Tensor.scalar(5))
  }

  it should "be identity op on a vector" in {
    Tensor.vector(1, 2, 3).const.transpose.eval should be(Tensor.vector(1, 2, 3))
  }

  it should "transpose a matrix" in {
    Tensor.matrix(Array(1, 2), Array(3, 4)).const.transpose.eval should be(Tensor.matrix(Array(1, 3), Array(2, 4)))
  }

  it should "transpose 3D tensor with custom permutatios" in {
    // todo: add a method to make 3D tensor
    val before = Tensor(range(1, 13), Shape(2, 2, 3))
    val after = Tensor(Array(1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12), Shape(2, 3, 2))
    before.const.transpose(Seq(0, 2, 1)).eval should be(after)
  }

  it should "support grad on a vector" in {
    val x = Tensor.vector(1, 2, 3).const
    x.transpose.sum.grad(x).returns[Float].eval should be(Tensor.vector(1, 1, 1))
  }

  it should "support grad on a matrix" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.transpose.sum.grad(x).returns[Float].eval should be(Tensor.matrix(Array(1, 1, 1), Array(1, 1, 1)))
  }

  "decaying average" should "work" in {
    10.0.const.decayingAvg(5.0.const, 0.9.const).eval should be(Tensor.scalar(9.5))
  }

  "rms" should "calculate root mean squared" in {
    val x = 8f.const
    val epsilon = 1f.const
    x.rms(epsilon).eval should be(Tensor.scalar(3f))
  }

  "boost" should "increase value on low iterations" in {
    val x = 1.5f.const
    val rate = 0.5f.const
    val iter = 2.const
    x.boost(rate, iter).eval should be(Tensor.scalar(2))
  }

  it should "have low effect on high iteration" in {
    val x = 1.5f.const
    val rate = 0.5f.const
    val iter = 1000000.const
    x.boost(rate, iter).eval should be(Tensor.scalar(1.5))  }

  "abs" should "return absolute value" in {
    Tensor.vector(-1, 2, -3).const.abs.eval should be(Tensor.vector(1, 2, 3))
  }

  "sigmoid" should "return a result of sigmoid function" in {
    Tensor.vector(1.0f, 5.0f, 10.0f).const.sigmoid.eval should
      be(Tensor.vector(0.7310586f, 0.9933072f, 0.9999546f))
  }

  "log" should "return a result of natural logarithm" in {
    Tensor.vector(1.0f, 5.0f, 10.0f).const.log.eval should
      be(Tensor.vector(0.0f, 1.609438f, 2.3025851f))
  }

  "round" should "return element-wise integer closest to x" in {
    Tensor.vector(1.6f, 1.4f, -1.7f).const.round.eval should
      be(Tensor.vector(2f, 1f, -2f))
  }
}