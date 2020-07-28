package org.scanet.math

import org.scanet.core.Output.Grad
import org.scanet.core.TensorType.{DoubleTag, FloatTag}
import org.scanet.core.syntax._
import org.scanet.core.{Output, Shape, Tensor, TensorType}
import org.scanet.math.Numeric.syntax._
import simulacrum.{op, typeclass}

import scala.Ordering.Implicits._

@typeclass trait MathBaseOp[F[_]] {

  /** Add two tensors. Supports broadcasting.
   *
   * {{{
   * val a = Tensor.matrix(
   *   Array(1, 2),
   *   Array(1, 2))
   * val b = Tensor.vector(1, 2)
   * val c = Tensor.matrix(
   *   Array(2, 4),
   *   Array(2, 4))
   * (a.const plus b.const).eval should be(c)
   * }}}
   *
   * @param left side
   * @param right side
   * @tparam C type which can be converted into output
   * @return a result of addition
   */
  // todo: figure out why + operator is not resolved
  @op("+", alias = true)
  def plus[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

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

  /** Subtract two tensors. Supports broadcasting.
   *
   * {{{(2.const - Tensor.vector(5, 10).const).eval should be(Tensor.vector(-3, -8))}}}
   *
   * @param left side
   * @param right side
   * @tparam C type which can be converted into output
   * @return a result of subtraction
   */
  @op("-", alias = true)
  def minus[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

  /** Negate a tensor.
   *
   * {{{(Tensor.vector(1, 2, 3).const.negate.eval should be(Tensor.vector(-1, -2, -3))}}}
   *
   * @param out output to negate
   * @return a result of negation
   */
  // todo: figure out why unary_- operator is not resolved
  @op("unary_-", alias = true)
  def negate[A: TensorType: Numeric](out: F[A]): F[A]

  /** Element-wise division. Supports broadcasting.
   *
   * {{{(Tensor.vector(5, 10, 15).const /:/ 5.const).eval should be(Tensor.vector(1, 2, 3))}}}
   *
   * @param left side
   * @param right side
   * @return a result of division
   */
  @op("/", alias = true)
  def div[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

  /** Element-wise multiplication. Supports broadcasting.
   *
   * {{{Tensor.vector(1, 2, 3).const *:* 5.const).eval should be(Tensor.vector(5, 10, 15))}}}
   *
   * @param left side
   * @param right side
   * @return a result of multiplication
   */
  @op("*", alias = true)
  def multiply[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

  /** Raises the tensor to the power of the exponent
   *
   * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
   *
   * @param out tensor
   * @return tensor `^` exponent
   */
  def pow[A: TensorType: Numeric](out: F[A], exponent: Float): F[A] = pow(out, Tensor.scalar(exponent).const)

  /** Raises the tensor to the power of the exponent
   *
   * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
   *
   * @param out tensor
   * @return tensor `^` exponent
   */
  def pow[A: TensorType: Numeric](out: F[A], exponent: Output[Float]): F[A]

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

  /** Raises the tensor to the power of two
   *
   * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.sqr.eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
   *
   * @param out tensor
   * @return tensor `^` 2
   */
  def sqr[A: TensorType: Numeric](out: F[A]): F[A] = pow(out, 2.0f)

  /** Returns square root of the given tensor
   *
   * {{{Tensor.vector(1.0f, 4.0f, 9.0f).const.sqrt.eval should be(Tensor.vector(1.0f, 2.0f, 3.0f))}}}
   *
   * @param out tensor
   * @return  tensor `^` 0.5
   */
  def sqrt[A: TensorType: Numeric](out: F[A]): F[A]

  /** Returns square root of the given tensor which never reaches zero
   * It is equivalent to `(out + epsilon) ^ 0.5`.
   *
   * {{{8f.const.sqrtZeroSafe(1f.const).eval should be(Tensor.scalar(3f))}}}
   *
   * @return tensor
   */
  def sqrtZeroSafe[A: TensorType: Numeric](out: F[A], epsilon: F[A]): F[A]

  /** Computes the sum of elements across dimensions of a tensor.
   *
   * Reduces `out` along the dimensions given in `axises`.
   * The rank of the tensor is reduced by 1 for each entry in `axises`.
   *
   * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum(Seq(0)).eval should be(Tensor.vector(5, 7, 9))}}}
   *
   * @param out tensor
   * @param axises to sum
   * @return tensor with summed values
   */
  def sum[A: TensorType: Numeric](out: F[A], axises: Seq[Int]): F[A]

  /** Computes the sum of elements across all dimensions of a tensor.
   *
   * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum.eval should be(Tensor.scalar(21))}}}
   *
   * @param out tensor
   * @return tensor with summed values
   */
  def sum[A: TensorType: Numeric](out: F[A]): F[A]

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
  def sigmoid[A: TensorType: Numeric](out: F[A]): F[A]

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

object MathBaseOp {

  trait Instances {
    implicit def outputIsMathOp: MathBaseOp[Output] = new OutputIsMathBaseOp
  }

  trait Syntax extends Instances with MathBaseOp.ToMathBaseOpOps with MathBaseStandaloneOps

  object syntax extends Syntax
}

class OutputIsMathBaseOp extends MathBaseOp[Output] with MathBaseStandaloneOps {

  override def plus[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot add tensors with shapes ${left.shape} + ${rightOut.shape}")
    Output.name[A]("Add")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val parentShape = parentGrad.shape
          val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
          val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
          List(
            sum(parentGrad, shrinkLeftAxises).reshape(left.shape),
            sum(parentGrad, shrinkRightAxises).reshape(rightOut.shape))
        }
      })
      .build
  }

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

  override def minus[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot subtracted tensors with shapes ${left.shape} - ${rightOut.shape}")
    Output.name[A]("Sub")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val parentShape = parentGrad.shape
          val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
          val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
          List(
            sum(parentGrad, shrinkLeftAxises).reshape(left.shape),
            sum(negate(parentGrad), shrinkRightAxises).reshape(rightOut.shape))
        }
      })
      .compileWithAllInputs
      .build
  }

  override def negate[A: TensorType: Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Neg")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          List(negate(parentGrad))
        }
      })
      .compileWithAllInputs
      .build
  }

  override def div[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot divide tensors with shapes ${left.shape} / ${rightOut.shape}")
    Output.name[A]("Div")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val parentShape = parentGrad.shape
          val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
          val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
          List(
            sum(div(parentGrad, rightOut.cast[R]), shrinkLeftAxises).reshape(left.shape),
            sum(
              negate(div(
                multiply(left.cast[R], parentGrad),
                multiply(rightOut, rightOut).cast[R]
              )),
              shrinkRightAxises
            ).reshape(rightOut.shape)
          )
        }
      })
      .compileWithAllInputs
      .build
  }

  override def pow[A: TensorType : Numeric](out: Output[A], exponent: Output[Float]): Output[A] = {
    Output.name[A]("Pow")
      .shape(out.shape)
      .inputs(out, exponent.as("exponent").cast[A])
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val p = pow(out.cast[R], minus(exponent, 1f.const))
          val local = multiply(p, exponent.cast[R])
          List(multiply(local, parentGrad))
        }
      })
      .compileWithAllInputs
      .build
  }

  override def exp[A: TensorType : Numeric: Floating](out: Output[A]) = {
    Output.name[A]("Exp")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          List(multiply(exp(out).cast[R], parentGrad))
        }
      })
      .compileWithAllInputs
      .build
  }

  override def sqrt[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Sqrt")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val local = multiply(pow(out.cast[R], -0.5f), 0.5f.const.cast[R])
          List(multiply(local, parentGrad))
        }
      })
      .compileWithAllInputs
      .build
  }

  override def sqrtZeroSafe[A: TensorType : Numeric](out: Output[A], epsilon: Output[A]): Output[A] = {
    sqrt(plus(out, epsilon))
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
            List(div(multiply(ones[R](out.shape), parentGrad.reshape(parentShape)), size.const.cast[R]))
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
    plus(multiply(decay, avg), multiply(minus(1.0f.const.cast[A], decay), next))
  }

  override def boost[A: TensorType : Numeric](out: Output[A], rate: Output[A], iter: Output[Int]): Output[A] = {
    div(out, minus(Numeric[A].one.const, pow(rate, iter.cast[Float])))
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

  override def sigmoid[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    val tag = TensorType[A].tag
    // todo: we need to introduce Floating (floating point) typeclass to type-check this
    require(tag == FloatTag || tag == DoubleTag, "only Float or Double is supported")
    Output.name[A]("Sigmoid")
      .shape(out.shape)
      .inputs(out)
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val s = sigmoid(out)
          val grad = multiply(s, minus(1.0f.const.cast[A], s))
          List(multiply(grad.cast[R], parentGrad))
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
          List(div(parentGrad, out.cast[R]))
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

trait MathBaseStandaloneOps {

  def plus[A: TensorType](ops: Output[A]*): Output[A] = {
    if (ops.size == 1) {
      ops.head
    } else {
      val shapes = ops.map(_.shape)
      require(shapes.distinct.size == 1, s"shapes of all tensors should be the same, but was ${shapes.mkString(" + ")}")
      Output.name[A]("AddN")
        .shape(shapes.head)
        .inputs(ops: _*)
        .compileWithInputList
        .localGrad(new Grad[A] {
          override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
            List.fill(ops.size)(parentGrad)
          }
        })
        .build
    }
  }

  def zeros[A: TensorType : Numeric](shape: Int*): Output[A] = fill(shape: _*)(Numeric[A].zero)

  def zeros[A: TensorType : Numeric](shape: Shape): Output[A] = fill(shape)(Numeric[A].zero)

  def ones[A: TensorType : Numeric](shape: Int*): Output[A] = fill(shape: _*)(Numeric[A].one)

  def ones[A: TensorType : Numeric](shape: Shape): Output[A] = fill(shape)(Numeric[A].one)

  def multiply[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot multiply tensors with shapes ${left.shape} * ${rightOut.shape}")
    Output.name[A]("Mul")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .localGrad(new Grad[A] {
        override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
          val parentShape = parentGrad.shape
          val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
          val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
          List(
            sum(multiply(rightOut.cast[R], parentGrad), shrinkLeftAxises).reshape(left.shape),
            sum(multiply(left.cast[R], parentGrad), shrinkRightAxises).reshape(rightOut.shape)
          )
        }
      })
      .build
  }

  def sum[A: TensorType : Numeric](out: Output[A], axises: Seq[Int]): Output[A] = {
    if (out.isScalar || axises.isEmpty) {
      out
    } else {
      Output.name[A]("Sum")
        .shape(out.shape.remove(axises: _*))
        .inputs(out, Tensor.vector(axises.map(_.toLong) :_*).const.as("axises"))
        .localGrad(new Grad[A] {
          override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
            // we need to recover reduced axises with 1, cause broadcasting will not always work
            val parentShape = axises.foldLeft(parentGrad.shape)((s, axis) => s.insert(axis, 1))
            List(multiply(ones[R](out.shape), parentGrad.reshape(parentShape)))
          }
        })
        .compileWithAllInputs
        .build
    }
  }

  def sum[A: TensorType : Numeric](out: Output[A]): Output[A] = sum(out, 0 until out.rank)

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
            multiply((left > right).cast[R], parentGrad),
            multiply((left < right).cast[R], parentGrad),
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
            multiply((left < right).cast[R], parentGrad),
            multiply((left > right).cast[R], parentGrad),
          )
        }
      })
      .compileWithAllInputs
      .build
  }
}
