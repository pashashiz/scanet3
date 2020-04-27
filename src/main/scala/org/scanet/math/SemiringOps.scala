package org.scanet.math

import org.scanet.core.{Output, Shape, TfType}
import org.scanet.core.CoreOps.syntax._

object SemiringOps {

  trait Instances {
    implicit def semiringOps[A: TfType: Semiring]: Semiring[Output[A]] = new SemiringOps[A]
  }

  trait Syntax extends Instances with Semiring.ToSemiringOps

  object syntax extends Syntax
}

class SemiringOps[A: TfType: Semiring] extends Semiring[Output[A]] {

  /** Add two elements.
    *
    * Requirements:
    * - `left` and `right` should have the same dimensions
    * - or one of the tensors should have shape which includes the other
    *
    * If both elements are scalars a simple scalar addition is done:
    *
    * For numbers
    * {{{2 plus 3 should be(5)}}}
    *
    * For tensors
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
    * @tparam B type which can be converted into [[ Output[A] ]]
    * @return a result of addition
    */
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

  /** Multiply 2 elements.
    *
    * If both elements are scalars a simple scalar multiplication is done:
    *
    * For numbers
    * {{{2 * 3 should be(6)}}}
    *
    * For tensors
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
    * @tparam B type which can be converted into [[ Output[A] ]]
    * @return a result of multiplication
    */
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
}
