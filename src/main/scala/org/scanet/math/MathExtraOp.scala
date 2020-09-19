package org.scanet.math

import org.scanet.core.Output.Grad
import org.scanet.core.syntax._
import org.scanet.core.{Output, Shape, Tensor, TensorType}
import org.scanet.math.MathBaseOp.syntax._
import org.scanet.math.Numeric.syntax._
import simulacrum.typeclass

import scala.Ordering.Implicits._

@typeclass trait MathExtraOp[F[_]] {

  /** Multiply 2 matrices. Example:
   * {{{
   * val a = Tensor.matrix(
   *       Array(1, 2, 3),
   *       Array(1, 2, 3))
   * val b = Tensor.matrix(
   *       Array(1, 2),
   *       Array(1, 2),
   *       Array(1, 2))
   * val c = Tensor.matrix(
   *       Array(6, 12),
   *       Array(6, 12))
   * (a.const * b.const).eval should be(c)
   * }}}
   *
   * NOTE: N-dim tensors are not supported yet, but that will be done in the future
   *
   * @param left side
   * @param right side
   * @tparam C type which can be converted into output
   * @return a result of multiplication
   */
  def matmul[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

  /**
   * Computes exponential of a given tensor element wise.
   *
   * {x.exp} is equal to {e.pow(x)}, however, exponential function is preferred cause it has
   * optimized kernel and identity derivative.
   *
   * `e` denotes Euler's number and is approximately equal to `2.718281`.
   *
   * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.exp.eval should be(Tensor.vector(2.7182817f, 7.389056f, 20.085537f))}}}
   *
   * @param out tensor
   * @return  exponent `^` tensor
   */
  def exp[A: TensorType : Numeric : Floating](out: F[A]): F[A]

  /** Computes the mean of elements across dimensions of a tensor.
   *
   * Reduces `out` along the dimensions given in `axises`.
   * The rank of the tensor is reduced by 1 for each entry in `axises`.
   *
   * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.mean(Seq(0)).eval should be(Tensor.vector(5, 7, 9))}}}
   *
   * @param out tensor
   * @param axises to sum
   * @return tensor with mean values
   */
  def mean[A: TensorType: Numeric](out: F[A], axises: Seq[Int]): F[A]

  /** Computes the mean of elements across all dimensions of a tensor.
   *
   * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.mean.eval should be(Tensor.scalar(21))}}}
   *
   * @param out tensor
   * @return tensor with mean values
   */
  def mean[A: TensorType: Numeric](out: F[A]): F[A]

  /** Shuffle dimensions of `out` according to a permutation.
   *
   * {{{
   * val before = Tensor(range(1, 13), Shape(2, 2, 3))
   * val after = Tensor(Array(1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12), Shape(2, 3, 2))
   * before.const.transpose(Seq(0, 2, 1)).eval should be(after)
   * }}}
   *
   * @param out tensor to transpose
   * @param perm dimensions permutations
   * @return transposed tensor
   */
  def transpose[A: TensorType: Numeric](out: F[A], perm: Seq[Int]): F[A]

  /** Transpose given tensor - flip dimensions over its diagonal.
   *
   * {{{
   * val matrix = Tensor.matrix(Array(1, 2), Array(3, 4)).const
   * matrix.transpose.eval should be(Tensor.matrix(Array(1, 3), Array(2, 4)))
   * }}}
   *
   * @param out tensor to transpose
   * @return transposed tensor
   */
  def transpose[A: TensorType: Numeric](out: F[A]): F[A]

  /** Compute decaying average of given value by putting more weight into latest (`next`) value.
   * It is equivalent to `(prev * decay) + ((1 - decay) * next)`.
   *
   * {{{10.0.const.decayingAvg(5.0.const, 0.9.const).eval should be(Tensor.scalar(9.5))}}}
   *
   * @param avg previous value
   * @param next current value
   * @param decay rate of decay
   * @return decaying average
   */
  def decayingAvg[A: TensorType: Numeric](avg: F[A], next: F[A], decay: F[A]): F[A]

  /** Increase (boost) given tensor on low values of `iter`.
   * Will return original value when `iter` approaches infinity.
   * It is equivalent to `out / (1 - rate ^ iter)`
   *
   * {{{
   * val x = 1.5f.const
   * val rate = 0.5f.const
   * val iter = 2.const
   * x.boost(rate, iter).eval should be(Tensor.scalar(2))
   * }}}
   *
   * @param out tensor
   * @param rate boost rate, should be < 1
   * @param iter computation iteration
   * @return unbiased value
   */
  def boost[A: TensorType: Numeric](out: F[A], rate: F[A], iter: F[Int]): F[A]

  /** Returns tensor with absolute values
   *
   * {{{Tensor.vector(-1, 2, -3).abs.eval should be(Tensor.vector(1, 2, 3)}}}
   *
   * @param out tensor
   * @return |tensor|
   */
  def abs[A: TensorType: Numeric](out: F[A]): F[A]

  /**
   * Computes sigmoid of `x` element-wise. Specifically
   *
   * {{{y = 1 / (1 + e^(-x))}}}
   *
   * @param out tensor
   * @return sigmoid
   */
  def sigmoid[A: TensorType: Numeric: Floating](out: F[A]): F[A]

  /**
   * Computes hyperbolic tangent of `x` element-wise.
   *
   * Given an input tensor, this function computes hyperbolic tangent of every
   * element in the tensor. Input range is `[-inf, inf]` and output range is `[-1,1]`.
   *
   * {{{Tensor.scalar(0.5f).const.tanh.eval should be(Tensor.scalar(0.46211717f))}}}
   *
   * @param out tensor
   * @return tanh
   */
  def tanh[A: TensorType: Numeric: Floating](out: F[A]): F[A]

  /**
   * Computes natural logarithm of `x` element-wise
   *
   * @param out tensor
   * @return natural logarithm
   */
  def log[A: TensorType: Numeric](out: F[A]): F[A]

  /**
   * Computes element-wise integer closest to `x`.
   *
   * @param out tensor
   * @return rounded tensor
   */
  def round[A: TensorType: Numeric](out: F[A]): F[A]
}

object MathExtraOp {

  trait Instances {
    implicit def outputIsMathExtraOp: MathExtraOp[Output] = new OutputIsMathExtraOp
  }

  trait Syntax extends Instances with MathExtraOp.ToMathExtraOpOps with MathExtraStandaloneOps

  object syntax extends Syntax
}

class OutputIsMathExtraOp extends MathExtraOp[Output] with MathExtraStandaloneOps {

  override def matmul[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.rank == 2 && rightOut.rank == 2,
      s"rank cannot be > 2 but got tensors with shapes ${left.shape} * ${rightOut.shape}")
    require(left.shape.last == rightOut.shape.head,
      s"cannot multiply tensors with shapes ${left.shape} * ${rightOut.shape}")
    val resultShape = Shape(left.shape.head, rightOut.shape.last)
    Output.name[A]("MatMul")
      .shape(resultShape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          List(
            matmul(parentGrad, transpose(rightOut).cast[R]),
            matmul(transpose(left).cast[R], parentGrad))
        }
      })
      .build
  }

  override def exp[A: TensorType : Numeric: Floating](out: Output[A]) = {
    Output.name[A]("Exp")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          List(exp(out).cast[R] * parentGrad)
        }
      })
      .compileWithAllInputs
      .build
  }

  override def mean[A: TensorType : Numeric](out: Output[A], axises: Seq[Int]): Output[A] = {
    if (out.isScalar || axises.isEmpty) {
      out
    } else {
      Output.name[A]("Mean")
        .shape(out.shape.remove(axises: _*))
        .inputs(out, Tensor.vector(axises.map(_.toLong) :_*).const.as("axises"))
        .localGrad(new Grad[A] {
          override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
            // we need to recover reduced axises with 1, cause broadcasting will not always work
            val parentShape = axises.foldLeft(parentGrad.shape)((s, axis) => s.insert(axis, 1))
            val size = out.shape.select(axises: _*).power
            List(ones[R](out.shape) * parentGrad.reshape(parentShape) / size.const.cast[R])
          }
        })
        .compileWithAllInputs
        .build
    }
  }

  override def mean[A: TensorType : Numeric](out: Output[A]): Output[A] = mean(out, 0 until out.rank)

  override def transpose[A: TensorType : Numeric](out: Output[A], perm: Seq[Int]): Output[A] = {
    if (out.isScalar) {
      out
    } else {
      Output.name[A]("Transpose")
        .shape(out.shape.permute(perm: _*))
        .inputs(out, Tensor.vector(perm.map(_.toLong) :_*).const.as("perm"))
        .localGrad(new Grad[A] {
          override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
            List(transpose(parentGrad))
          }
        })
        .compileWithAllInputs
        .build
    }
  }

  override def transpose[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    transpose(out, (0 until out.rank).reverse)
  }

  override def decayingAvg[A: TensorType : Numeric](avg: Output[A], next: Output[A], decay: Output[A]): Output[A] = {
    decay * avg plus (1.0f.const.cast[A] - decay) * next
  }

  override def boost[A: TensorType : Numeric](out: Output[A], rate: Output[A], iter: Output[Int]): Output[A] = {
    out / (Numeric[A].one.const - rate.pow(iter.cast[Float]))
  }

  override def abs[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Abs")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          List(abs(parentGrad))
        }
      })
      .compileWithAllInputs
      .build
  }

  override def sigmoid[A: TensorType : Numeric: Floating](out: Output[A]): Output[A] = {
    Output.name[A]("Sigmoid")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val s = sigmoid(out)
          val grad = s * (1.0f.const.cast[A] - s)
          List(grad.cast[R] * parentGrad)
        }
      })
      .compileWithAllInputs
      .build
  }

  override def tanh[A: TensorType : Numeric : Floating](out: Output[A]) = {
    Output.name[A]("Tanh")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val grad = 1f.const.cast[A] - tanh(out).sqr
          List(grad.cast[R] * parentGrad)
        }
      })
      .compileWithAllInputs
      .build
  }

  override def log[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Log")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          List(parentGrad / out.cast[R])
        }
      })
      .compileWithAllInputs
      .build
  }

  override def round[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Rint")
      .shape(out.shape)
      .inputs(out)
      .compileWithAllInputs
      .build
  }
}

trait MathExtraStandaloneOps {

  def max[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot compare tensors with shapes ${left.shape} <> ${rightOut.shape}")
    Output.name[A]("Maximum")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          import org.scanet.math.MathLogicalOp.syntax._
          List(
            (left > right).cast[R] * parentGrad,
            (left < right).cast[R] * parentGrad,
          )
        }
      })
      .compileWithAllInputs
      .build
  }

  def min[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot compare tensors with shapes ${left.shape} <> ${rightOut.shape}")
    Output.name[A]("Minimum")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          import org.scanet.math.MathLogicalOp.syntax._
          List(
            (left < right).cast[R] * parentGrad,
            (left > right).cast[R] * parentGrad,
          )
        }
      })
      .compileWithAllInputs
      .build
  }
}
