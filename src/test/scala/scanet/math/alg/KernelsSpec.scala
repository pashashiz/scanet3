package scanet.math.alg

import scanet.test.CustomMatchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Expr, Shape, Tensor}
import scanet.math.syntax._

import scala.collection.immutable.Seq

class KernelsSpec extends AnyWordSpec with CustomMatchers {

  "zeros" should {
    "fill a tensor with zeros" in {
      zeros[Int](2).eval should be(Tensor.vector(0, 0))
    }
  }

  "ones" should {
    "fill a tensor with ones" in {
      ones[Int](2).eval should be(Tensor.vector(1, 1))
    }
  }

  "const" should {

    "have ones gradient if input is same const" in {
      val a = 2.const
      (a grad a).returns[Float].eval should be(Tensor.scalar(1))
    }

    "fail to find a gradient if input is not a part of the computation graph" in {
      the[IllegalArgumentException] thrownBy {
        val a = 2.const
        val b = 3.const
        (a grad b).returns[Float]
      } should have message "requirement failed: " +
      "cannot find a gradient with respect to Const(3)[Int]:() cause that input is not a part of the computation graph"
    }
  }

  "plus" should {

    "add 2 scalars" in {
      (2.0f.const plus 5.0f.const).eval should be(Tensor.scalar(7.0f))
    }

    "add 2 tensors when one includes shape of the other" in {
      val a = Tensor.matrix(Array(1, 2), Array(1, 2))
      val b = Tensor.vector(1, 2)
      val c = Tensor.matrix(Array(2, 4), Array(2, 4))
      (a.const plus b.const).eval should be(c)
    }

    "work when adding same tensor" in {
      val a = 5.0f.const.as("a")
      (a plus a).as("c").eval should be(Tensor.scalar(10.0f))
    }

    "fail when 2 tensors have incompatible dimensions" in {
      the[IllegalArgumentException] thrownBy {
        Tensor.matrix(Array(1, 2), Array(1, 2)).const plus Tensor.vector(1, 2, 3).const
      } should have message
      "requirement failed: cannot add tensors with shapes (2, 2) + (3)"
    }

    "calculate a gradient if left side is a differentiable variable" in {
      val a = 3.const
      val x = 2.const
      ((x + a) grad x).returns[Float].eval should be(Tensor.scalar(1))
    }

    "calculate a gradient if right side is a differentiable variable" in {
      val a = 2.const
      val x = 3.const
      ((a + x) grad x).returns[Float].eval should be(Tensor.scalar(1))
    }

    "calculate a gradient if right and left side is a differentiable variable" in {
      val x = 2.const
      ((x + x) grad x).returns[Float].eval should be(Tensor.scalar(2))
    }

    "calculate a gradient for minus op if left side is a differentiable variable" in {
      val a = 3.const
      val x = 2.const
      ((x - a) grad x).returns[Float].eval should be(Tensor.scalar(1))
    }

    "calculate a gradient for minus op if right side is a differentiable variable" in {
      val a = 2.const
      val x = 3.const
      ((a - x) grad x).returns[Float].eval should be(Tensor.scalar(-1))
    }

    "calculate a gradient for minus op if right and left side is a differentiable variable" in {
      val x = 2.const
      ((x - x) grad x).returns[Float].eval should be(Tensor.scalar(0))
    }

    "calculate a gradient with broadcasting when smaller tensor is a differentiable variable" in {
      val a = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val x = Tensor.vector(1, 2, 3).const
      val grad = Tensor.vector(2, 2, 2)
      ((a + x).sum grad x).returns[Float].eval should be(grad)
      ((x + a).sum grad x).returns[Float].eval should be(grad)
    }

    "calculate a gradient with broadcasting when bigger tensor is a differentiable variable" in {
      val x = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val a = Tensor.vector(1, 2, 3).const
      val grad = Tensor.matrix(Array(1, 1, 1), Array(1, 1, 1))
      ((x + a).sum grad x).returns[Float].eval should be(grad)
      ((a + x).sum grad x).returns[Float].eval should be(grad)
    }
  }

  "plus N" should {

    "add multiple tensors" in {
      plus(1.const, 2.const, 3.const).eval should be(Tensor.scalar(6))
    }

    "fail when tensors have different shapes" in {
      the[IllegalArgumentException] thrownBy {
        plus(1.const, 2.const, Tensor.vector(3, 4).const).eval
      } should have message
      "requirement failed: shapes of all tensors should be the same, but was () + () + (2)"
    }

    "calculate gradient for plus N" in {
      val a = 1.const
      val b = 2.const
      val x = 3.const
      plus(a, b, x).sum.grad(x).returns[Float].eval should be(Tensor.scalar(1))
    }

    "calculate gradient for plus N when diff variable is used more than once" in {
      val a = Tensor.vector(1, 2).const
      val x = Tensor.vector(5, 6).const
      plus(a, x, x).sum.grad(x).returns[Float].eval should be(Tensor.vector(2, 2))
    }
  }

  "minus" should {

    "subtract 2 scalars" in {
      (5.const - 3.const).eval should be(Tensor.scalar(2))
    }

    "subtract 2 tensors with same shape" in {
      (Tensor.vector(5, 10).const - Tensor.vector(1, 2).const).eval should be(Tensor.vector(4, 8))
    }

    "subtract 2 tensors when left includes shape of right" in {
      (Tensor.vector(5, 10).const - 2.const).eval should be(Tensor.vector(3, 8))
    }

    "subtract 2 tensors when right includes shape of left" in {
      (2.const - Tensor.vector(5, 10).const).eval should be(Tensor.vector(-3, -8))
    }

    "fail to subtract when 2 tensors have incompatible dimensions" in {
      the[IllegalArgumentException] thrownBy {
        (Tensor.matrix(Array(1, 2), Array(1, 2)).const - Tensor.vector(1, 2, 3).const).eval
      } should have message "requirement failed: cannot subtracted tensors with shapes (2, 2) - (3)"
    }

    "calculate a gradient with broadcasting when smaller tensor is a differentiable variable" in {
      val a = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val x = Tensor.vector(1, 2, 3).const
      ((a - x).sum grad x).returns[Float].eval should be(Tensor.vector(-2, -2, -2))
      ((x - a).sum grad x).returns[Float].eval should be(Tensor.vector(2, 2, 2))
    }

    "calculate a gradient with broadcasting when bigger tensor is a differentiable variable" in {
      val x = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val a = Tensor.vector(1, 2, 3).const
      ((x - a).sum grad x).returns[Float].eval should be(
        Tensor.matrix(Array(1, 1, 1), Array(1, 1, 1)))
      ((a - x).sum grad x).returns[Float].eval should be(
        Tensor.matrix(Array(-1, -1, -1), Array(-1, -1, -1)))
    }
  }

  "negate" should {

    "work on a scalar" in {
      3.const.negate.eval should be(Tensor.scalar(-3))
    }

    "support gradient negation for scalars" in {
      val x: Expr[Int] = 3.const
      x.negate.grad(x).returns[Float].eval should be(Tensor.scalar(-1))
    }

    "work on a tensor" in {
      Tensor.vector(1, 2, 3).const.negate.eval should be(Tensor.vector(-1, -2, -3))
    }

    "support gradient of negation for vectors" in {
      val x = Tensor.vector(1, 2, 3).const
      x.negate.sum.grad(x).returns[Float].eval should be(Tensor.vector(-1, -1, -1))
    }
  }

  "element-wise division" should {

    "work on tensors with same dimensions" in {
      (Tensor.vector(5, 10, 15).const / Tensor.vector(5, 5, 5).const).eval should be(
        Tensor.vector(1, 2, 3))
    }

    "work for floating point numbers" in {
      (Tensor.vector(2.0f, 4.0f, 6.0f).const / Tensor
        .vector(10.0f, 10.0f, 10.0f)
        .const).eval should be(Tensor.vector(0.2f, 0.4f, 0.6f))
    }

    "support broadcasting with vector divided by scalar" in {
      (Tensor.vector(5, 10, 15).const / 5.const).eval should be(Tensor.vector(1, 2, 3))
    }

    "support broadcasting with matrix divided by matrix with last dimension 1" in {
      import scanet.math.alg.kernels.syntax._
      val left: Tensor[Int] = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30))
      val right: Tensor[Int] = Tensor.matrix(Array(1), Array(5))
      val expected = Tensor.matrix(Array(5, 10, 15), Array(4, 5, 6))
      (left.const / right.const).eval should be(expected)
    }

    "fail when tensors have incompatible dimensions" in {
      the[IllegalArgumentException] thrownBy {
        (Tensor.matrix(Array(1, 2), Array(1, 2)).const / Tensor.vector(1, 2, 3).const).eval
      } should have message "requirement failed: cannot divide tensors with shapes (2, 2) / (3)"
    }

    "calculate gradient for constants" in {
      val a = 4.const
      val x = 2.const

      (a div x).sum.grad(x).returns[Float].eval should be(Tensor.scalar(-1.0))
      (a div x).sum.grad(a).returns[Float].eval should be(Tensor.scalar(0.5))
    }

    "calculate gradient for vectors" in {
      val a = Tensor.vector(5, 10, 15).const
      val x = Tensor.vector(5, 5, 5).const

      (a div x).sum.grad(x).returns[Float].eval should be(Tensor.vector(-0.2f, -0.4f, -0.6f))
      (a div x).sum.grad(a).returns[Float].eval should be(Tensor.vector(0.2f, 0.2f, 0.2f))
    }

    "calculate gradient for matrices with broadcasting when smaller tensor is differentiable value" in {
      val a = Tensor.matrix(Array(12, 16, 20), Array(24, 28, 32)).const
      val x = Tensor.vector(2, 4, 8).const

      (a div x).sum.grad(x).returns[Float].eval should be(Tensor.vector(-9f, -2.75f, -0.8125f))
      (x div a).sum.grad(x).returns[Float].eval should be(
        Tensor.vector(0.125f, 0.09821428f, 0.08125f))
    }

    "calculate gradient for matrices with broadcasting when bigger tensor is differentiable value" in {
      val a = Tensor.matrix(Array(12, 16, 20), Array(24, 28, 32)).const
      val x = Tensor.vector(2, 4, 8).const

      (a div x).sum.grad(a).returns[Float].eval should be(
        Tensor.matrix(Array(0.5f, 0.25f, 0.125f), Array(0.5f, 0.25f, 0.125f)))
      val grad = Tensor.matrix(
        Array(-0.013888889f, -0.015625f, -0.02f),
        Array(-0.0034722222f, -0.0051020407f, -0.0078125f))
      (x div a).sum.grad(a).returns[Float].eval should be(grad)
    }

    "calculate gradient for matrices with broadcasting second matrix has last dimension 1" in {
      val a = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val x = Tensor.matrix(Array(1), Array(5)).const
      (a div x).sum.grad(x).returns[Float].eval should be(
        Tensor.matrix(Array(-30.0f), Array(-3.0f)))
    }
  }

  "element-wise multiplication" should {

    "work on tensors with same dimensions" in {
      (Tensor.vector(1, 2, 3).const * Tensor.vector(5, 5, 5).const).eval should be(
        Tensor.vector(5, 10, 15))
    }

    "support broadcasting" in {
      (Tensor.vector(1, 2, 3).const * 5.const).eval should be(Tensor.vector(5, 10, 15))
    }

    "fail when tensors have incompatible dimensions" in {
      the[IllegalArgumentException] thrownBy {
        (Tensor.matrix(Array(1, 2), Array(1, 2)).const * Tensor.vector(1, 2, 3).const).eval
      } should have message "requirement failed: cannot multiply tensors with shapes (2, 2) * (3)"
    }

    "calculate a gradient equals to right side if left side is a differentiable variable" in {
      val a = 3.const
      val x = 2.const
      ((x * a) grad x).returns[Float].eval should be(Tensor.scalar(3))
    }

    "calculate a gradient equals to left side if right side is a differentiable variable" in {
      val a = 3.const
      val x = 2.const
      ((a * x) grad x).returns[Float].eval should be(Tensor.scalar(3))
    }

    "calculate a gradient equals to sum if right and left side is a differentiable variable" in {
      val x = Tensor.scalar(3).const
      ((x * x) grad x).returns[Float].eval should be(Tensor.scalar(6))
    }

    "x" in {
      val x = Tensor.matrix(Array(1, 2, 3)).const
      val a = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      println((a * x).eval)
    }

    "calculate a gradient with full dimension broadcasting when smaller tensor is a differentiable variable" in {
      val a = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val x = Tensor.vector(1, 2, 3).const
      val grad = Tensor.vector(25, 35, 45)
      ((a * x).sum grad x).returns[Float].eval should be(grad)
      ((x * a).sum grad x).returns[Float].eval should be(grad)
    }

    "calculate a gradient with full dimension broadcasting broadcasting when bigger tensor is a differentiable variable" in {
      val x = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val a = Tensor.vector(1, 2, 3).const
      val grad = Tensor.matrix(Array(1, 2, 3), Array(1, 2, 3))
      ((x * a).sum grad x).returns[Float].eval should be(grad)
      ((a * x).sum grad x).returns[Float].eval should be(grad)
    }

    "calculate a gradient with broadcasting of dimensions with size 1 when smaller tensor is a differentiable variable" in {
      val a = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val x = Tensor.matrix(Array(1, 2, 3)).const
      val grad = Tensor.matrix(Array(25.0f, 35.0f, 45.0f))
      ((a * x).sum grad x).returns[Float].eval should be(grad)
    }

    "calculate a gradient with broadcasting of dimensions with size 1 when bigger tensor is a differentiable variable" in {
      val x = Tensor.matrix(Array(5, 10, 15), Array(20, 25, 30)).const
      val a = Tensor.matrix(Array(1, 2, 3)).const
      val grad = Tensor.matrix(Array(1.0f, 2.0f, 3.0f), Array(1.0f, 2.0f, 3.0f))
      ((a * x).sum grad x).returns[Float].eval should be(grad)
    }
  }

  "pow" should {

    "compute the power of the tensor with high exponent" in {
      Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))
    }

    "return same tensor if exponent is 1" in {
      Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(1).eval should be(Tensor.vector(1.0f, 2.0f, 3.0f))
    }

    "return ones tensor if exponent is 0" in {
      Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(0).eval should be(Tensor.vector(1.0f, 1.0f, 1.0f))
    }

    "calculate a gradient" in {
      val x = Tensor.vector(5, 10, 15).const
      x.pow(3).sum.grad(x).returns[Float].eval should be(Tensor.vector(75.0f, 300.0f, 675.0f))
    }
  }

  "sqrt" should {

    "compute square root of tensor" in {
      Tensor.vector(1.0f, 4.0f, 9.0f).const.sqrt.eval should be(Tensor.vector(1.0f, 2.0f, 3.0f))
    }

    "calculate a gradient" in {
      //  grad = dy * 0.5 / y, where y = sqrt
      val x = Tensor.vector(1.0f, 4.0f, 16.0f).const
      x.sqrt.sum.grad(x).returns[Float].eval should be(Tensor.vector(0.5f, 0.25, 0.125f))
    }
  }

  "rsqrt" should {

    "compute reciprocal root of tensor" in {
      Tensor.vector(1.0f, 4.0f, 9.0f).const.rsqrt.roundAt(3).eval should be(
        Tensor.vector(1.0f, 0.5f, 0.333f))
    }

    "calculate a gradient" in {
      val x = Tensor.vector(1.0f, 4.0f, 16.0f).const
      x.rsqrt.sum.grad(x).returns[Float].roundAt(3).eval should be(
        Tensor.vector(-0.5f, -0.062f, -0.008f))
    }
  }

  "sum" should {

    "calculate sum across all axis by default" in {
      Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum.eval should be(Tensor.scalar(21))
    }

    "support reducing along matrix columns" in {
      Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum(Seq(0)).eval should be(
        Tensor.vector(5, 7, 9))
    }

    "support reducing along matrix rows" in {
      Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum(Seq(1)).eval should be(
        Tensor.vector(6, 15))
    }

    "support reducing 4D tensors" in {
      val tensor = Tensor(Array(1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4), Shape(2, 2, 2, 2))
      tensor.const.sum(Seq(0, 1)).eval should be(Tensor.matrix(Array(1, 2), Array(3, 4)))
    }
  }

  "matmul" should {

    "multiply 2 matrices" in {
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
      (a.const matmul b.const).eval should be(c)
    }

    "fail to multiply tensors with rank < 2" in {
      the[IllegalArgumentException] thrownBy {
        (Tensor.vector(1, 2).const matmul Tensor.matrix(Array(1, 2), Array(3, 4)).const).eval
      } should have message "requirement failed: rank cannot be < 2 " +
      "but got tensors with shapes (2) * (2, 2)"
    }

    "fail to multiply tensors with incompatible shapes" in {
      the[IllegalArgumentException] thrownBy {
        val a = Tensor(Array.range(0, 12), Shape(2, 2, 3))
        val b = Tensor(Array.range(0, 8), Shape(2, 2, 2))
        (a.const matmul b.const).eval
      } should have message "requirement failed: cannot matmul tensors with shapes (2, 2, 3) * (2, 2, 2)"
    }

    "fail to multiply tensors with not broadcastable batch dimensions" in {
      the[IllegalArgumentException] thrownBy {
        val a = Tensor(Array.range(0, 8), Shape(2, 2, 2))
        val b = Tensor(Array.range(0, 12), Shape(3, 2, 2))
        (a.const matmul b.const).eval
      } should have message "requirement failed: cannot matmul tensors with shapes (2, 2, 2) * (3, 2, 2)"
    }

    "multiply 2 batches of matrices" in {
      val a = Tensor(Array(
        Array(Array(1, 2), Array(3, 4)),
        Array(Array(5, 6), Array(7, 8))))
      val c = Tensor(Array(
        Array(Array(7, 10), Array(15, 22)),
        Array(Array(67, 78), Array(91, 106))))
      (a.const matmul a.const).eval should be(c)
    }

    "multiply a batch of matrices by a matrix" in {
      val a = Tensor(Array(
        Array(Array(1, 2), Array(3, 4)),
        Array(Array(5, 6), Array(7, 8))))
      val w = Tensor(Array(Array(1, 2), Array(3, 4)))
      val c = Tensor(Array(
        Array(Array(7, 10), Array(15, 22)),
        Array(Array(23, 34), Array(31, 46))))
      (a.const matmul w.const).eval should be(c)
    }

    "multiply a matrix by a batch of matrices" in {
      val w = Tensor(Array(Array(1, 2), Array(3, 4)))
      val a = Tensor(Array(
        Array(Array(1, 2), Array(3, 4)),
        Array(Array(5, 6), Array(7, 8))))
      val c = Tensor(Array(
        Array(Array(7, 10), Array(15, 22)),
        Array(Array(19, 22), Array(43, 50))))
      (w.const matmul a.const).eval should be(c)
    }

    "multiply batches with broadcast of a dimension of size 1" in {
      val a = Tensor(Array(Array(Array(1, 2), Array(3, 4))))
      val b = Tensor(Array(
        Array(Array(1, 0), Array(0, 1)),
        Array(Array(2, 0), Array(0, 2))))
      val c = Tensor(Array(
        Array(Array(1, 2), Array(3, 4)),
        Array(Array(2, 4), Array(6, 8))))
      (a.const matmul b.const).eval should be(c)
    }

    "multiply 4D tensors with non square matrices" in {
      val a = Tensor(Array.fill(2 * 3 * 4 * 5)(1), Shape(2, 3, 4, 5))
      val b = Tensor(Array.fill(2 * 3 * 5 * 6)(1), Shape(2, 3, 5, 6))
      val c = (a.const matmul b.const).eval
      c.shape should be(Shape(2, 3, 4, 6))
      c.toArray.forall(_ == 5) shouldBe true
    }

    "have correct gradient when 2 batches of matrices are given" in {
      val a = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2)).const
      val x = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2)).const
      // parent grad (ones) matmul x.transpose: rows are summed up
      val gradLeft = Tensor(Array(3, 7, 3, 7, 11, 15, 11, 15), Shape(2, 2, 2))
      // a.transpose matmul parent grad (ones): columns are summed up
      val gradRight = Tensor(Array(4, 4, 6, 6, 12, 12, 14, 14), Shape(2, 2, 2))
      ((a matmul x).sum grad x).returns[Float].eval should be(gradRight)
      ((x matmul a).sum grad x).returns[Float].eval should be(gradLeft)
    }

    "have correct gradient when a batch of matrices is multiplied by a matrix" in {
      val a = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2)).const
      val w = Tensor.matrix(Array(1, 2), Array(3, 4)).const
      // batch dimension is broadcast for the matrix so grads are summed up over batches
      val gradW = Tensor.matrix(Array(16, 16), Array(20, 20))
      val gradA = Tensor(Array(3, 7, 3, 7, 3, 7, 3, 7), Shape(2, 2, 2))
      ((a matmul w).sum grad w).returns[Float].eval should be(gradW)
      ((a matmul w).sum grad a).returns[Float].eval should be(gradA)
    }

    "have correct gradient when a matrix is multiplied by a batch of matrices" in {
      val w = Tensor.matrix(Array(1, 2), Array(3, 4)).const
      val a = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2)).const
      // sum over batches of ones matmul a.transpose: batch 0 -> (3, 7), batch 1 -> (11, 15)
      val gradW = Tensor.matrix(Array(14, 22), Array(14, 22))
      // w.transpose matmul ones: columns of w are summed up for each batch
      val gradA = Tensor(Array(4, 4, 6, 6, 4, 4, 6, 6), Shape(2, 2, 2))
      ((w matmul a).sum grad w).returns[Float].eval should be(gradW)
      ((w matmul a).sum grad a).returns[Float].eval should be(gradA)
    }

    "have correct gradient when a batch dimension of size 1 is broadcast" in {
      val a = Tensor(Array(1, 2, 3, 4), Shape(1, 2, 2)).const
      val b = Tensor(Array(1, 0, 0, 1, 2, 0, 0, 2), Shape(2, 2, 2)).const
      // ones matmul b.transpose for each batch is summed up: (1, 1) + (2, 2) per row
      ((a matmul b).sum grad a).returns[Float].eval should be(
        Tensor(Array(3, 3, 3, 3), Shape(1, 2, 2)))
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
      // parent grad matmul a.transpose
      // |1 1| x |5  15 25| = |15, 35, 55|
      // |1 1|   |10 20 30|   |15, 35, 55|
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

    "calculate mean across all axis by default" in {
      val tensor = Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const
      tensor.mean.eval should be(Tensor.scalar(3.5f))
    }

    "support reducing along matrix columns" in {
      val tensor = Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const
      tensor.mean(Seq(0)).eval should be(Tensor.vector(2.5f, 3.5f, 4.5f))
    }

    "support reducing along matrix rows" in {
      val tensor = Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const
      tensor.mean(Seq(1)).eval should be(Tensor.vector(2f, 5f))
    }

    "reduce and keep dimensions" in {
      val tensor = Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const
      tensor.mean(Seq(1), keepDims = true).eval should be(Tensor.matrix(Array(2f), Array(5f)))
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
      val y = x.mean(Seq(0))
      val grad = Tensor.matrix(Array(0.5f, 0.5f, 0.5f), Array(0.5f, 0.5f, 0.5f))
      y.sum.grad(x).returns[Float].eval should be(grad)
    }

    "have correct gradient and keep dimensions" in {
      val x = Tensor.matrix(Array(1f, 2f, 3f), Array(4f, 5f, 6f)).const
      val y = x.mean(Seq(0), keepDims = true)
      val grad = Tensor.matrix(Array(0.5f, 0.5f, 0.5f), Array(0.5f, 0.5f, 0.5f))
      y.sum.grad(x).returns[Float].eval should be(grad)
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
      val before = Tensor(Array.range(1, 13), Shape(2, 2, 3))
      val after = Tensor(Array(1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12), Shape(2, 3, 2))
      before.const.transpose(Seq(0, 2, 1)).eval should be(after)
    }

    "support grad on 3D tensor with custom permutations" in {
      val x = Tensor(Array.range(0, 24).map(_.toFloat), Shape(2, 3, 4)).const
      val w = Tensor(Array.range(0, 24).map(i => (i * i % 7).toFloat), Shape(3, 4, 2)).const
      // grad of a weighted sum is the weights permuted back
      (x.transpose(Seq(1, 2, 0)) * w).sum.grad(x).returns[Float].eval should be(
        w.transpose(Seq(2, 0, 1)).eval)
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
      Tensor.vector(1.0f, 5.0f, 10.0f).const.sigmoid.roundAt(6).eval should
      be(Tensor.vector(0.731059f, 0.993307f, 0.999955f))
    }
  }

  "tanh" should {

    "return a result of tanh function" in {
      Tensor.vector(-0.5f, 0f, 0.5f).const.tanh.eval should
      beCloseTo(Tensor.vector(-0.46211717f, 0.0f, 0.46211717f))
    }

    "have correct gradient" in {
      val x = Tensor.vector(-0.5f, 0f, 0.5f).const
      x.tanh.sum.grad(x).returns[Float].eval should
      beCloseTo(Tensor.vector(0.7864477f, 1.0f, 0.7864477f))
    }
  }

  "sign" should {
    "return an element-wise indication of the sign of a number" in {
      val x = Tensor.vector(-0.5f, 0f, 0.5f).const
      x.sign.eval shouldBe Tensor.vector(-1f, 0f, 1f)
    }
  }
}
