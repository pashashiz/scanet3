package org.scanet.math.alg

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import org.scanet.core.{Shape, Tensor}
import org.scanet.math.syntax._

import scala.Array.range
import scala.collection.immutable.Seq

class MathExtraOpSpec extends AnyWordSpec with Matchers {

  "matmul" should {
    "multiply 2 matrices" in {
      val a = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6))
      val b = Tensor.matrix(Array(1, 2), Array(1, 2), Array(1, 2))
      val c = Tensor.matrix(Array(6, 12), Array(15, 30))
      (a.const matmul b.const).eval should be(c)
    }
    "fail to multiply 3D tensors" in {
      the[IllegalArgumentException] thrownBy {
        val tensor = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2))
        (tensor.const matmul tensor.const).eval
      } should have message "requirement failed: rank cannot be > 2 but got tensors with shapes (2, 2, 2) * (2, 2, 2)"
    }
    "have correct gradient when 2 matrices are given and right side is a differentiable variable" in {
      val a = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
      val x = Tensor.matrix(Array(5, 10), Array(15, 20), Array(25, 30)).const
      val grad = Tensor.matrix(Array(5, 5), Array(7, 7), Array(9, 9))
      ((a matmul x).sum grad x).returns[Float].eval should be(grad)
    }
    "have correct gradient when 2 matrices are given and left side is a differentiable variable" in {
      val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
      val a = Tensor.matrix(Array(5, 10), Array(15, 20), Array(25, 30)).const
      val grad = Tensor.matrix(Array(15, 35, 55), Array(15, 35, 55))
      ((x matmul a).sum grad x).returns[Float].eval should be(grad)
    }
  }

  "exp" should {
    "compute the exponent of a tensor element wise" in {
      Tensor.vector(1.0f, 2.0f, 3.0f).const.exp.eval should be(
        Tensor.vector(2.7182817f, 7.389056f, 20.085537f))
    }
    "have identity gradient" in {
      val x = Tensor.vector(1.0f, 2.0f, 3.0f).const
      x.exp.sum.grad(x).returns[Float].eval should be(
        Tensor.vector(2.7182817f, 7.389056f, 20.085537f))
    }
  }

  "mean" should {
    "calculate mean across all axises by default" in {
      Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const.mean.eval should be(
        Tensor.scalar(3.5f))
    }
    "support reducing along matrix columns" in {
      Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const.mean(Seq(0)).eval should be(
        Tensor.vector(2.5f, 3.5f, 4.5f))
    }
    "support reducing along matrix rows" in {
      Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const.mean(Seq(1)).eval should be(
        Tensor.vector(2f, 5f))
    }
    "support reducing 4D tensors" in {
      val tensor = Tensor(
        Array(1f, 0f, 0f, 0f, 0f, 2f, 0f, 0f, 0f, 0f, 3f, 0f, 0f, 0f, 0f, 4f),
        Shape(2, 2, 2, 2))
      tensor.const.mean(Seq(0, 1)).eval should be(
        Tensor.matrix(Array(0.25f, 0.5f), Array(0.75f, 1f)))
    }
    "have correct gradient" in {
      val x = Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const
      val grad = Tensor.matrix(Array(0.5f, 0.5f, 0.5f), Array(0.5f, 0.5f, 0.5f))
      x.mean(Seq(0)).sum.grad(x).returns[Float].eval should be(grad)
    }
  }

  "transpose" should {
    "be identity op on a scalar" in {
      Tensor.scalar(5).const.transpose.eval should be(Tensor.scalar(5))
    }
    "be identity op on a vector" in {
      Tensor.vector(1, 2, 3).const.transpose.eval should be(Tensor.vector(1, 2, 3))
    }
    "transpose a matrix" in {
      Tensor.matrix(Array(1, 2), Array(3, 4)).const.transpose.eval should be(
        Tensor.matrix(Array(1, 3), Array(2, 4)))
    }
    "transpose 3D tensor with custom permutatios" in {
      // todo: add a method to make 3D tensor
      val before = Tensor(range(1, 13), Shape(2, 2, 3))
      val after = Tensor(Array(1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12), Shape(2, 3, 2))
      before.const.transpose(Seq(0, 2, 1)).eval should be(after)
    }
    "support grad on a vector" in {
      val x = Tensor.vector(1, 2, 3).const
      x.transpose.sum.grad(x).returns[Float].eval should be(Tensor.vector(1, 1, 1))
    }
    "support grad on a matrix" in {
      val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
      x.transpose.sum.grad(x).returns[Float].eval should be(
        Tensor.matrix(Array(1, 1, 1), Array(1, 1, 1)))
    }
  }

  "decaying average" should {
    "work" in {
      10.0.const.decayingAvg(5.0.const, 0.9.const).eval should be(Tensor.scalar(9.5))
    }
  }

  "sqrtZeroSafe" should {
    "calculate square root with epsilon" in {
      val x = 8f.const
      val epsilon = 1f.const
      x.sqrtZeroSafe(epsilon).eval should be(Tensor.scalar(3f))
    }
  }

  "boost" should {
    "increase value on low iterations" in {
      val x = 1.5f.const
      val rate = 0.5f.const
      val iter = 2.const
      x.boost(rate, iter).eval should be(Tensor.scalar(2))
    }
    "have low effect on high iteration" in {
      val x = 1.5f.const
      val rate = 0.5f.const
      val iter = 1000000.const
      x.boost(rate, iter).eval should be(Tensor.scalar(1.5))
    }
  }

  "abs" should {
    "return absolute value" in {
      Tensor.vector(-1, 2, -3).const.abs.eval should be(Tensor.vector(1, 2, 3))
    }
  }

  "log" should {
    "return a result of natural logarithm" in {
      Tensor.vector(1.0f, 5.0f, 10.0f).const.log.eval should
      be(Tensor.vector(0.0f, 1.609438f, 2.3025851f))
    }
  }

  "round" should {
    "return element-wise integer closest to x" in {
      Tensor.vector(1.6f, 1.4f, -1.7f).const.round.eval should
      be(Tensor.vector(2f, 1f, -2f))
    }
  }

  "max" should {
    "return maximum value" in {
      max(Tensor.vector(1, 2, 3).const, Tensor.vector(4, 5, 6).const).eval should be(
        Tensor.vector(4, 5, 6))
    }
    "support broadcasting" in {
      max(Tensor.vector(1, 2).const, Tensor.vector(4).const).eval should be(Tensor.vector(4, 4))
    }
    "propagate gradient if derived against max branch" in {
      val left = Tensor.vector(1, 5).const
      val right = Tensor.vector(3, 4).const
      val y = max(left, right).sum
      (y grad left).returns[Float].eval should be(Tensor.vector(0f, 1f))
    }
  }

  "min" should {
    "return minimum value" in {
      min(Tensor.vector(1, 2, 3).const, Tensor.vector(4, 5, 6).const).eval should be(
        Tensor.vector(1, 2, 3))
    }
    "propagate gradient if derived against min branch" in {
      val left = Tensor.vector(1, 5).const
      val right = Tensor.vector(3, 4).const
      val y = min(left, right).sum
      (y grad left).returns[Float].eval should be(Tensor.vector(1f, 0f))
    }
  }

  "sigmoid" should {
    "return a result of sigmoid function" in {
      Tensor.vector(1.0f, 5.0f, 10.0f).const.sigmoid.eval should
      be(Tensor.vector(0.7310586f, 0.9933071f, 0.9999546f))
    }
  }

  "tanh" should {
    "return a result of tanh function" in {
      Tensor.vector(-0.5f, 0f, 0.5f).const.tanh.eval should
      be(Tensor.vector(-0.46211717f, 0.0f, 0.46211717f))
    }
    "have correct gradient" in {
      val x = Tensor.vector(-0.5f, 0f, 0.5f).const
      x.tanh.sum.grad(x).returns[Float].eval should
      be(Tensor.vector(0.7864477f, 1.0f, 0.7864477f))
    }
  }
}
