package org.scanet.math

import org.scanet.core.{Output, Shape, TfType}
import org.scanet.core.CoreOp.syntax._
import simulacrum.{op, typeclass}

@typeclass trait MathOp[A] {

  /** Add two tensors.
   *
   * Requirements:
   * - `left` and `right` should have the same dimensions
   * - or one of the tensors should have shape which includes the other
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
   * @tparam B type which can be converted into output
   * @return a result of addition
   */
  // todo: figure out why + operator is not resolved
  @op("+", alias = true)
  def plus[B](left: A, right: B)(implicit c: Convertible[B, A]): A

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
   * @tparam B type which can be converted into output
   * @return a result of multiplication
   */
  @op("*", alias = true)
  def multiply[B](left: A, right: B)(implicit c: Convertible[B, A]): A

  /** Subtract two tensors.
   *
   * Requirements:
   * - `left` and `right` should have the same dimensions
   * - or one of the tensors should have shape which includes the other
   *
   * {{{(2.const - Tensor.vector(5, 10).const).eval should be(Tensor.vector(-3, -8))}}}
   *
   * @param left side
   * @param right side
   * @tparam B type which can be converted into output
   * @return a result of subtraction
   */
  @op("-", alias = true)
  def minus[B](left: A, right: B)(implicit c: Convertible[B, A]): A

  /** Negate a tensor.
   *
   * {{{(Tensor.vector(1, 2, 3).const.negate.eval should be(Tensor.vector(-1, -2, -3))}}}
   *
   * @param out output to negate
   * @return a result of negation
   */
  // todo: figure out why unary_- operator is not resolved
  @op("unary_-", alias = true)
  def negate(out: A): A
}

object MathOp {

  trait Instances {
    implicit def outputIsMathOp[A: TfType: Numeric]: MathOp[Output[A]] = new OutputIsMathOp[A]
  }

  trait Syntax extends Instances with MathOp.ToMathOpOps

  object syntax extends Syntax
}

class OutputIsMathOp[A: TfType: Numeric] extends MathOp[Output[A]] {

  override def plus[B](left: Output[A], right: B)(implicit c: Convertible[B, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.shape.endsWith(rightOut.shape) || rightOut.shape.endsWith(left.shape) ,
      s"tensors with shapes ${left.shape} and ${rightOut.shape} cannot be added, " +
        "one of the tensors should have shape which includes the other")
    Output.name("Add")
      .shape(if (left.rank > rightOut.rank) left.shape else rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def multiply[B](left: Output[A], right: B)(implicit c: Convertible[B, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    val leftAdjusted = left.reshape(left.shape.alignLeft(2, using = 1))
    val rightAdjusted = rightOut.reshape(rightOut.shape.alignLeft(2, using = 1))
    require(leftAdjusted.rank == 2 && rightAdjusted.rank == 2,
      s"rank cannot be > 2 but got tensors with shapes ${leftAdjusted.shape} * ${rightAdjusted.shape}")
    require(leftAdjusted.shape.last == rightAdjusted.shape.head,
      s"cannot multiply tensors with shapes ${leftAdjusted.shape} * ${rightAdjusted.shape}")
    val resultShape = Shape(leftAdjusted.shape.head, rightAdjusted.shape.last)
    val result = Output.name("MatMul")
      .shape(resultShape)
      .inputs(leftAdjusted, rightAdjusted)
      .compileWithAllInputs
      .build
    // we need to prune additional adjusted dimensions added for scalars and vectors
    val adjusted = 2 - math.min(left.shape.rank, rightOut.shape.rank)
    result.reshape(resultShape.prune(adjusted))
  }

  override def minus[B](left: Output[A], right: B)(implicit c: Convertible[B, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.shape.endsWith(rightOut.shape) || rightOut.shape.endsWith(left.shape) ,
      s"tensors with shapes ${left.shape} and ${rightOut.shape} cannot be subtracted, " +
        "one of the tensors should have shape which includes the other")
    Output.name("Sub")
      .shape(if (left.rank > rightOut.rank) left.shape else rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def negate(a: Output[A]): Output[A] = {
    Output.name("Neg")
      .shape(a.shape)
      .inputs(a)
      .compileWithAllInputs
      .build
  }
}
