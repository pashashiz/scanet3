package org.scanet.math.logical

import org.scanet.core
import org.scanet.core.syntax._
import org.scanet.core._
import org.scanet.math.Logical.syntax._
import org.scanet.math.Numeric.syntax._
import org.scanet.math.{Convertible, Logical}

import scala.collection.immutable.Seq
import scala.math.Ordering.Implicits._

case class All[A: TensorType: Logical](expr: Expr[A], axises: Seq[Int]) extends Expr[Boolean] {
  override def name: String = "All"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = expr.shape.remove(axises: _*)
  override val inputs: Seq[Expr[_]] = Seq(expr, Tensor.vector(axises.map(_.toLong): _*).const)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class Any[A: TensorType: Logical](expr: Expr[A], axises: Seq[Int]) extends Expr[Boolean] {
  override def name: String = "Any"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = expr.shape.remove(axises: _*)
  override val inputs: Seq[Expr[_]] = Seq(expr, Tensor.vector(axises.map(_.toLong): _*).const)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class Equal[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot check for equality tensors with shapes ${left.shape} :== ${right.shape}")
  override def name: String = "Equal"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class NotEqual[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot check for non-equality tensors with shapes ${left.shape} :!= ${right.shape}")
  override def name: String = "NotEqual"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class Greater[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot compare tensors with shapes ${left.shape} > ${right.shape}")
  override def name: String = "Greater"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class GreaterEqual[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot compare tensors with shapes ${left.shape} >= ${right.shape}")
  override def name: String = "GreaterEqual"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class Less[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot compare tensors with shapes ${left.shape} < ${right.shape}")
  override def name: String = "Less"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class LessEqual[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot compare tensors with shapes ${left.shape} <= ${right.shape}")
  override def name: String = "LessEqual"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class And[A: TensorType: Logical](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot logically AND tensors with shapes ${left.shape} && ${right.shape}")
  override def name: String = "LogicalAnd"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class Or[A: TensorType: Logical](left: Expr[A], right: Expr[A]) extends Expr[Boolean] {
  require(
    left.broadcastableAny(right),
    s"cannot logically OR tensors with shapes ${left.shape} || ${right.shape}")
  override def name: String = "LogicalOr"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

case class Not[A: TensorType: Logical](expr: Expr[A]) extends Expr[Boolean] {
  override def name: String = "LogicalNot"
  override def tpe: Option[TensorType[Boolean]] = Some(TensorType[Boolean])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: core.Compiler[Boolean] = DefaultCompiler[Boolean]()
}

trait AllKernels {

  def all[A: TensorType: Logical](expr: Expr[A]): Expr[Boolean] = all(expr, 0 until expr.rank)
  def all[A: TensorType: Logical](expr: Expr[A], axises: Seq[Int]): Expr[Boolean] =
    All(expr, axises)

  def any[A: TensorType: Logical](expr: Expr[A]): Expr[Boolean] = any(expr, 0 until expr.rank)
  def any[A: TensorType: Logical](expr: Expr[A], axises: Seq[Int]): Expr[Boolean] =
    Any(expr, axises)

  def eq[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = {
    val rightExpr: Expr[A] = c.convert(right)
    require(
      left.shape == rightExpr.shape,
      s"cannot check equality tensors with different shapes ${left.shape} === ${rightExpr.shape}")
    all(eqElementWise(left, right))
  }

  def neq[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = {
    val rightExpr: Expr[A] = c.convert(right)
    require(
      left.shape == rightExpr.shape,
      s"cannot check non-equality tensors with different shapes ${left.shape} !== ${rightExpr.shape}")
    any(neqElementWise(left, right))
  }

  def eqElementWise[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = Equal(left, c.convert(right))

  def neqElementWise[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = NotEqual(left, c.convert(right))

  def gt[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = Greater(left, c.convert(right))

  def gte[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = GreaterEqual(left, c.convert(right))

  def lt[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = Less(left, c.convert(right))

  def lte[A: TensorType, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = LessEqual(left, c.convert(right))

  def and[A: TensorType: Logical, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = And(left, c.convert(right))

  def or[A: TensorType: Logical, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = Or(left, c.convert(right))

  def xor[A: TensorType: Logical, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[Boolean] = {
    val rightExpr: Expr[A] = c.convert(right)
    require(
      left.broadcastableAny(rightExpr),
      s"cannot logically XOR tensors with shapes ${left.shape} ^ ${rightExpr.shape}")
    // (x | y) & (~x | ~y);
    and(or(left, rightExpr), or(not(left), not(rightExpr)))
  }

  def not[A: TensorType: Logical](expr: Expr[A]): Expr[Boolean] = Not(expr)

}

object kernels extends AllKernels {

  class AnyOps[A: TensorType](expr: Expr[A]) {
    import org.scanet.math.logical.{kernels => f}

    /** Check if 2 tensors are equal (tensors should have the same shape)
      *
      * {{{(Tensor.vector(1, 2, 3).const === Tensor.vector(1, 2, 3).const).eval should be(Tensor.scalar(true))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def eq(right: Expr[A]): Expr[Boolean] = f.eq(expr, right)
    def ===(right: Expr[A]): Expr[Boolean] = f.eq(expr, right)

    /** Check if 2 tensors are not equal (tensors should have the same shape)
      *
      * {{{(Tensor.vector(1, 2, 3).const !== Tensor.vector(1, 2, 4).const).eval should be(Tensor.scalar(true))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def neq(right: Expr[A]): Expr[Boolean] = f.neq(expr, right)
    def !==(right: Expr[A]): Expr[Boolean] = f.neq(expr, right)

    /** Element-wise equality check. Supports broadcasting.
      *
      * {{{(Tensor.vector(1, 2, 3).const :== Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(true, true, false))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def eqElementWise(right: Expr[A]): Expr[Boolean] = f.eqElementWise(expr, right)
    def :==(right: Expr[A]): Expr[Boolean] = f.eqElementWise(expr, right)

    /** Element-wise non-equality check. Supports broadcasting.
      *
      * {{{(Tensor.vector(1, 2, 3).const :!= Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(false, false, true))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def neqElementWise(right: Expr[A]): Expr[Boolean] = f.neqElementWise(expr, right)
    def :!=(right: Expr[A]): Expr[Boolean] = f.neqElementWise(expr, right)

    /** Element-wise check whether left tensor is greater than right tensor. Supports broadcasting.
      *
      * {{{(Tensor.vector(1, 2, 3).const > Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(false, false, true))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def gt(right: Expr[A]): Expr[Boolean] = f.gt(expr, right)
    def >(right: Expr[A]): Expr[Boolean] = f.gt(expr, right)

    /** Element-wise check whether left tensor is greater or equal than right tensor. Supports broadcasting.
      *
      * {{{(Tensor.vector(1, 2, 3).const >= Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(false, true, true))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def gte(right: Expr[A]): Expr[Boolean] = f.gte(expr, right)
    def >=(right: Expr[A]): Expr[Boolean] = f.gte(expr, right)

    /** Element-wise check whether left tensor is less than right tensor. Supports broadcasting.
      *
      * {{{(Tensor.vector(1, 2, 3).const < Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(true, false, false))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def lt(right: Expr[A]): Expr[Boolean] = f.lt(expr, right)
    def <(right: Expr[A]): Expr[Boolean] = f.lt(expr, right)

    /** Element-wise check whether left tensor is less or equal than right tensor. Supports broadcasting.
      *
      * {{{(Tensor.vector(1, 2, 3).const <= Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(true, true, false))}}}
      *
      * @param right side
      * @return a result of the check
      */
    def lte(right: Expr[A]): Expr[Boolean] = f.lte(expr, right)
    def <=(right: Expr[A]): Expr[Boolean] = f.lte(expr, right)
  }

  class LogicalOps[A: TensorType: Logical](expr: Expr[A]) {
    import org.scanet.math.logical.{kernels => f}

    /** Reduces tensor along the dimensions given in axis using logical AND.
      *
      * {{{Tensor.matrix(Array(true, false), Array(true, true)).const.all(Seq(1)).eval should be(Tensor.vector(false, true))}}}
      *
      * @param axises axises to make a reduction
      * @return a result of reduction
      */
    def all(axises: Seq[Int]): Expr[Boolean] = f.all(expr, axises)

    /** Reduces tensor along all dimensions using logical AND.
      *
      * {{{Tensor.vector(true, false).const.all.eval should be(Tensor.scalar(false))}}}
      *
      * @return a result of reduction
      */
    def all: Expr[Boolean] = f.all(expr, 0 until expr.rank)

    /** Reduces tensor along the dimensions given in axis using logical OR.
      *
      * {{{Tensor.matrix(Array(false, false), Array(true, true)).const.any(Seq(1)).eval should be(Tensor.vector(false, true))}}}
      *
      * @param axises axises to make a reduction
      * @return a result of reduction
      */
    def any(axises: Seq[Int]): Expr[Boolean] = f.any(expr, axises)

    /** Reduces tensor along all dimensions using logical OR.
      *
      * {{{Tensor.vector(true, false).const.any.eval should be(Tensor.scalar(true))}}}
      *
      * @return a result of reduction
      */
    def any: Expr[Boolean] = f.any(expr, 0 until expr.rank)

    /** Logically AND element-wise. Supports broadcasting.
      *
      * {{{(Tensor.vector(true, false).const && Tensor.vector(true, true).const).eval should be(Tensor.vector(true, false))}}}
      *
      * @param right side
      * @return a result of logical AND
      */
    def and(right: Expr[A]): Expr[Boolean] = f.and(expr, right)
    def &&(right: Expr[A]): Expr[Boolean] = f.and(expr, right)

    /** Logically OR element-wise. Supports broadcasting.
      *
      * {{{(Tensor.vector(true, false).const || Tensor.vector(true, true).const).eval should be(Tensor.vector(true, true))}}}
      *
      * @param right side
      * @return a result of logical OR
      */
    def or(right: Expr[A]): Expr[Boolean] = f.or(expr, right)
    def ||(right: Expr[A]): Expr[Boolean] = f.or(expr, right)

    /** Logically XOR element-wise. Supports broadcasting.
      *
      * {{{(Tensor.vector(true, false).const ^ Tensor.vector(true, true).const).eval should be(Tensor.vector(false, true))}}}
      *
      * @param right side
      * @return a result of logical XOR
      */
    def xor(right: Expr[A]): Expr[Boolean] = f.xor(expr, right)
    def ^(right: Expr[A]): Expr[Boolean] = f.xor(expr, right)

    /** Logically NOT element-wise.
      *
      * {{{Tensor.vector(true, false).const.not.eval should be(Tensor.vector(false, true))}}}
      *
      * @return a result of logical NOT
      */
    def not: Expr[Boolean] = f.not(expr)
    def unary_! : Expr[Boolean] = f.not(expr)
  }

  trait AllSyntax extends AllKernels {
    implicit def toLogicalKernelAnyOps[A: TensorType](expr: Expr[A]): AnyOps[A] =
      new AnyOps[A](expr)
    implicit def toLogicalKernelLogicalOps[A: TensorType: Logical](expr: Expr[A]): LogicalOps[A] =
      new LogicalOps[A](expr)
  }

  object syntax extends AllSyntax
}
