package org.scanet.math

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

  /** Multiply 2 tensors.
   *
   * If both elements are scalars a simple scalar multiplication is done:
   * {{{(2.const * 3.const).eval should be(Tensor.scalar(6))}}}
   *
   * If a left element is a scalar and right is a vector, each element in a vector will be multiplied by a scalar
   * {{{(2.const * Tensor.vector(1, 2, 3).const).eval should be(Tensor.vector(2, 4, 6))}}}
   *
   * If a left element is a vector and right is a scalar - an error will be raised cause dimensions are incompatible
   *
   * If a left element is a vector and right is a matrix, a vector will be reshaped into a matrix
   * and a matrix multiplication will be done, the result will be squeezed to a vector though
   * {{{
   *  val a = Tensor.vector(1, 2, 3)
   *  val b = Tensor.matrix(
   *       Array(1, 2),
   *       Array(1, 2),
   *       Array(1, 2))
   * (a.const * b.const).eval should be(Tensor.vector(6, 12))
   * }}}
   *
   * If two matrices are multiplied the regular matmul is done:
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
  @op("*", alias = true)
  def multiply[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

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
  @op(":*", alias = true)
  def multiplyElementWise[A: TensorType: Numeric, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[A]

  /** Raises the tensor to the power of the exponent
   *
   * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
   *
   * @param out tensor
   * @return tensor `^` exponent
   */
  def pow[A: TensorType: Numeric](out: F[A], exponent: Float): F[A]

  def sqrt[A: TensorType: Numeric](out: F[A]): F[A]

  def sum[A: TensorType: Numeric](out: F[A], axises: Seq[Int]): F[A]

  def sum[A: TensorType: Numeric](out: F[A]): F[A]

  def transpose[A: TensorType: Numeric](out: F[A], perm: Seq[Int]): F[A]

  def transpose[A: TensorType: Numeric](out: F[A]): F[A]
}

object MathBaseOp {

  trait Instances {
    implicit def outputIsMathOp: MathBaseOp[Output] = new OutputIsMathBaseOp
  }

  trait Syntax extends Instances with MathBaseOp.ToMathBaseOpOps with MathBaseMultiOp

  object syntax extends Syntax
}

class OutputIsMathBaseOp extends MathBaseOp[Output] {

  override def plus[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot add tensors with shapes ${left.shape} + ${rightOut.shape}")
    Output.name[A]("Add")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .localGrad(ctx => {
        val parentShape = ctx.parentGrad.shape
        val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
        val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
        List(
          sum(ctx.parentGrad, shrinkLeftAxises),
          sum(ctx.parentGrad, shrinkRightAxises))
      })
      .build
  }

  override def multiply[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    val leftAdjusted = left.reshape(left.shape.alignLeft(2, using = 1))
    val rightAdjusted = rightOut.reshape(rightOut.shape.alignLeft(2, using = 1))
    require(leftAdjusted.rank == 2 && rightAdjusted.rank == 2,
      s"rank cannot be > 2 but got tensors with shapes ${leftAdjusted.shape} * ${rightAdjusted.shape}")
    require(leftAdjusted.shape.last == rightAdjusted.shape.head,
      s"cannot multiply tensors with shapes ${leftAdjusted.shape} * ${rightAdjusted.shape}")
    val resultShape = Shape(leftAdjusted.shape.head, rightAdjusted.shape.last)
    val result = Output.name[A]("MatMul")
      .shape(resultShape)
      .inputs(leftAdjusted, rightAdjusted)
      .compileWithAllInputs
      .localGrad(ctx => {
        List(
          multiply(ctx.parentGrad, transpose(rightAdjusted).cast[Float]),
          multiply(transpose(leftAdjusted).cast[Float], ctx.parentGrad))
      })
      .build
    val adjusted = 2 - math.min(left.shape.rank, rightOut.shape.rank)
    result.reshape(resultShape.prune(adjusted))
  }

  override def minus[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot subtracted tensors with shapes ${left.shape} - ${rightOut.shape}")
    Output.name[A]("Sub")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .localGrad(ctx => {
        val parentShape = ctx.parentGrad.shape
        val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
        val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
        List(sum(ctx.parentGrad, shrinkLeftAxises), sum(negate(ctx.parentGrad), shrinkRightAxises))
      })
      .compileWithAllInputs
      .build
  }

  override def negate[A: TensorType: Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Neg")
      .shape(out.shape)
      .inputs(out)
      .localGrad(ctx => List(negate(ctx.parentGrad)))
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
      .localGrad(ctx => {
        val parentShape = ctx.parentGrad.shape
        val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
        val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
        List(
          sum(div(ctx.parentGrad, rightOut.cast[Float]), shrinkLeftAxises),
          sum(
            negate(div(
              multiplyElementWise(left.cast[Float], ctx.parentGrad),
              multiplyElementWise(rightOut, rightOut).cast[Float]
            )),
            shrinkRightAxises
          )
        )
      })
      .compileWithAllInputs
      .build
  }

  override def multiplyElementWise[A: TensorType: Numeric, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot multiply tensors with shapes ${left.shape} :* ${rightOut.shape}")
    Output.name[A]("Mul")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .localGrad(ctx => {
        val parentShape = ctx.parentGrad.shape
        val shrinkRightAxises = parentShape.broadcastableAxises(rightOut.shape)
        val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape)
        List(
          sum(multiplyElementWise(rightOut.cast[Float], ctx.parentGrad), shrinkLeftAxises),
          sum(multiplyElementWise(left.cast[Float], ctx.parentGrad), shrinkRightAxises)
        )
      })
      .build
  }

  override def pow[A: TensorType : Numeric](out: Output[A], exponent: Float): Output[A] = {
    Output.name[A]("Pow")
      .shape(out.shape)
      .inputs(out, Tensor.scalar(exponent).const.as("exponent").cast[A])
      .localGrad(ctx => {
        val local = multiplyElementWise(pow(out.cast[Float], exponent - 1), exponent.toFloat.const)
        List(multiplyElementWise(local, ctx.parentGrad))
      })
      .compileWithAllInputs
      .build
  }

  override def sqrt[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    Output.name[A]("Sqrt")
      .shape(out.shape)
      .inputs(out)
      .localGrad(ctx => {
        val local = multiplyElementWise(pow(out.cast[Float], -0.5f), 0.5f.const)
        List(multiplyElementWise(local, ctx.parentGrad))
      })
      .compileWithAllInputs
      .build
  }

  override def sum[A: TensorType : Numeric](out: Output[A], axises: Seq[Int]): Output[A] = {
    if (out.isScalar || axises.isEmpty) {
      out
    } else {
      Output.name[A]("Sum")
        .shape(out.shape.remove(axises: _*))
        .inputs(out, Tensor.vector(axises.map(_.toLong) :_*).const.as("axises"))
        .localGrad(ctx => List(multiplyElementWise(Tensor.ones[Float](out.shape).const, ctx.parentGrad)))
        .compileWithAllInputs
        .build
    }
  }

  override def sum[A: TensorType : Numeric](out: Output[A]): Output[A] = sum(out, 0 until out.rank)

  override def transpose[A: TensorType : Numeric](out: Output[A], perm: Seq[Int]): Output[A] = {
    if (out.isScalar) {
      out
    } else {
      Output.name[A]("Transpose")
        .shape(out.shape.permute(perm: _*))
        .inputs(out, Tensor.vector(perm.map(_.toLong) :_*).const.as("perm"))
        .localGrad(ctx => List(transpose(ctx.parentGrad)))
        .compileWithAllInputs
        .build
    }
  }

  override def transpose[A: TensorType : Numeric](out: Output[A]): Output[A] = {
    transpose(out, (0 until out.rank).reverse)
  }
}

trait MathBaseMultiOp {
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
        .localGrad(ctx => List.fill(ops.size)(ctx.parentGrad))
        .build
    }
  }
}
