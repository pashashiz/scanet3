package org.scanet.math

import org.scanet.core.Const.syntax._
import org.scanet.core.TensorType.syntax._
import org.scanet.core.{Output, Tensor, TensorType}
import simulacrum.{op, typeclass}

import scala.Ordering.Implicits._
import scala.language.higherKinds

@typeclass trait MathBoolOp[F[_]] {

  /** Check if 2 tensors are equal (tensors should have the same shape)
   *
   * {{{(Tensor.vector(1, 2, 3).const === Tensor.vector(1, 2, 3).const).eval should be(Tensor.scalar(true))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op("===", alias = true)
  def eq[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Check if 2 tensors are not equal (tensors should have the same shape)
   *
   * {{{(Tensor.vector(1, 2, 3).const !== Tensor.vector(1, 2, 4).const).eval should be(Tensor.scalar(true))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op("!==", alias = true)
  def neq[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Element-wise equality check. Supports broadcasting.
   *
   * {{{(Tensor.vector(1, 2, 3).const :== Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(true, true, false))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op(":==", alias = true)
  def eqElementWise[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Element-wise non-equality check. Supports broadcasting.
   *
   * {{{(Tensor.vector(1, 2, 3).const :!= Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(false, false, true))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op(":!=", alias = true)
  def neqElementWise[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Reduces tensor along the dimensions given in axis using logical AND.
   *
   * {{{Tensor.matrix(Array(true, false), Array(true, true)).const.all(Seq(1)).eval should be(Tensor.vector(false, true))}}}
   *
   * @param axises axises to make a reduction
   * @return a result of reduction
   */
  def all[A: TensorType](out: F[A], axises: Seq[Int]): F[Boolean]

  /** Reduces tensor along all dimensions using logical AND.
   *
   * {{{Tensor.vector(true, false).const.all.eval should be(Tensor.scalar(false))}}}
   *
   * @return a result of reduction
   */
  def all[A: TensorType](out: F[A]): F[Boolean]

  /** Reduces tensor along the dimensions given in axis using logical OR.
   *
   * {{{Tensor.matrix(Array(false, false), Array(true, true)).const.any(Seq(1)).eval should be(Tensor.vector(false, true))}}}
   *
   * @param axises axises to make a reduction
   * @return a result of reduction
   */
  def any[A: TensorType](out: F[A], axises: Seq[Int]): F[Boolean]

  /** Reduces tensor along all dimensions using logical OR.
   *
   * {{{Tensor.vector(true, false).const.any.eval should be(Tensor.scalar(true))}}}
   *
   * @return a result of reduction
   */
  def any[A: TensorType](out: F[A]): F[Boolean]

  /** Element-wise check whether left tensor is greater than right tensor. Supports broadcasting.
   *
   * {{{(Tensor.vector(1, 2, 3).const > Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(false, false, true))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op(">", alias = true)
  def gt[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Element-wise check whether left tensor is greater or equal than right tensor. Supports broadcasting.
   *
   * {{{(Tensor.vector(1, 2, 3).const >= Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(false, true, true))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op(">=", alias = true)
  def gte[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Element-wise check whether left tensor is less than right tensor. Supports broadcasting.
   *
   * {{{(Tensor.vector(1, 2, 3).const < Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(true, false, false))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op("<", alias = true)
  def lt[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

  /** Element-wise check whether left tensor is less or equal than right tensor. Supports broadcasting.
   *
   * {{{(Tensor.vector(1, 2, 3).const <= Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(true, true, false))}}}
   *
   * @param left side
   * @param right side
   * @return a result of the check
   */
  @op("<=", alias = true)
  def lte[A: TensorType, C](left: F[A], right: C)(implicit c: Convertible[C, F[A]]): F[Boolean]

}

object MathBoolOp {

  trait Instances {
    implicit def outputIsMathOpBool: MathBoolOp[Output] = new OutputIsMathBoolOp
  }

  trait Syntax extends Instances with MathBoolOp.ToMathBoolOpOps

  object syntax extends Syntax
}

class OutputIsMathBoolOp extends MathBoolOp[Output] {

  override def all[A: TensorType](out: Output[A]): Output[Boolean] = all(out, 0 until out.rank)

  override def all[A: TensorType](out: Output[A], axises: Seq[Int]): Output[Boolean] = {
    require(axises.forall(_ < out.rank), s"tensor with rank ${out.rank} does not have (${axises.mkString(", ")}) axises")
    Output.name[Boolean]("All")
      .shape(out.shape)
      .inputs(out, Tensor.vector(axises.map(_.toLong) :_*).const)
      .compileWithAllInputs
      .build
  }

  override def any[A: TensorType](out: Output[A]): Output[Boolean] = any(out, 0 until out.rank)

  override def any[A: TensorType](out: Output[A], axises: Seq[Int]): Output[Boolean] = {
    require(axises.forall(_ < out.rank), s"tensor with rank ${out.rank} does not have (${axises.mkString(", ")}) axises")
    Output.name[Boolean]("Any")
      .shape(out.shape)
      .inputs(out, Tensor.vector(axises.map(_.toLong) :_*).const)
      .compileWithAllInputs
      .build
  }

  override def eq[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.shape == rightOut.shape,
      s"cannot check equality tensors with different shapes ${left.shape} === ${rightOut.shape}")
    all(eqElementWise(left, right))
  }

  override def neq[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.shape == rightOut.shape,
      s"cannot check non-equality tensors with different shapes ${left.shape} !== ${rightOut.shape}")
    any(neqElementWise(left, right))
  }

  override def eqElementWise[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot check for equality tensors with shapes ${left.shape} :== ${rightOut.shape}")
    Output.name[Boolean]("Equal")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def neqElementWise[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot check for equality tensors with shapes ${left.shape} :== ${rightOut.shape}")
    Output.name[Boolean]("NotEqual")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def gt[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot compare tensors with shapes ${left.shape} > ${rightOut.shape}")
    Output.name[Boolean]("Greater")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def gte[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot compare tensors with shapes ${left.shape} >= ${rightOut.shape}")
    Output.name[Boolean]("GreaterEqual")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def lt[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot compare tensors with shapes ${left.shape} < ${rightOut.shape}")
    Output.name[Boolean]("Less")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def lte[A: TensorType, C](left: Output[A], right: C)(implicit c: Convertible[C, Output[A]]): Output[Boolean] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.broadcastableAny(rightOut),
      s"cannot compare tensors with shapes ${left.shape} <= ${rightOut.shape}")
    Output.name[Boolean]("LessEqual")
      .shape(left.shape max rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }
}
