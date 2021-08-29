package org.scanet.math.alg

import org.scanet.core
import org.scanet.core._
//import org.scanet.core.kernels._
import org.scanet.math.Numeric.syntax._
import org.scanet.math.logical.kernels.syntax._
import org.scanet.math.{Convertible, Floating, Numeric}
import org.scanet.core.Const.syntax._

import scala.collection.immutable.Seq
import scala.math.Ordering.Implicits._
import org.scanet.core.TensorType.syntax._
import org.scanet.core.kernels.syntax._

case class Plus[A: TensorType: Numeric](left: Expr[A], right: Expr[A]) extends Expr[A] {
  require(
    left.broadcastableAny(right),
    s"cannot add tensors with shapes ${left.shape} + ${right.shape}")
  override def name: String = "Add"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val parentShape = parentGrad.shape
      val shrinkRightAxises = parentShape.broadcastableAxises(right.shape).toList
      val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape).toList
      List(
        Sum(parentGrad, shrinkLeftAxises).reshape(left.shape),
        Sum(parentGrad, shrinkRightAxises).reshape(right.shape))
    }
  }
}

case class PlusN[A: TensorType: Numeric] private (expr: Seq[Expr[A]]) extends Expr[A] {
  override def name: String = "AddN"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.head.shape
  override def inputs: Seq[Expr[_]] = expr
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] =
      Seq.fill(expr.size)(parentGrad)
  }
}

object PlusN {
  def apply[A: TensorType: Numeric](expr: Seq[Expr[A]]): Expr[A] = {
    if (expr.size == 1) {
      expr.head
    } else {
      val shapes = expr.map(_.shape)
      require(
        shapes.distinct.size == 1,
        s"shapes of all tensors should be the same, but was ${shapes.mkString(" + ")}")
      new PlusN(expr)
    }
  }
}

case class Minus[A: TensorType: Numeric](left: Expr[A], right: Expr[A]) extends Expr[A] {
  require(
    left.broadcastableAny(right),
    s"cannot subtracted tensors with shapes ${left.shape} - ${right.shape}")
  override def name: String = "Sub"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val parentShape = parentGrad.shape
      val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape).toList
      val shrinkRightAxises = parentShape.broadcastableAxises(right.shape).toList
      List(
        Sum(parentGrad, shrinkLeftAxises).reshape(left.shape),
        Sum(Negate(parentGrad), shrinkRightAxises).reshape(right.shape))
    }
  }
}

case class Negate[A: TensorType: Numeric](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Neg"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] =
      List(Negate(parentGrad))
  }
}

case class Multiply[A: TensorType: Numeric] private (left: Expr[A], right: Expr[A])
    extends Expr[A] {
  require(
    left.broadcastableAny(right),
    s"cannot multiply tensors with shapes ${left.shape} * ${right.shape}")
  override def name: String = "Mul"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val parentShape = parentGrad.shape
      val shrinkRightAxises = parentShape.broadcastableAxises(right.shape).toList
      val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape).toList
      List(
        Sum(Multiply(right.cast[R], parentGrad), shrinkLeftAxises).reshape(left.shape),
        Sum(Multiply(left.cast[R], parentGrad), shrinkRightAxises).reshape(right.shape))
    }
  }
}

case class Pow[A: TensorType: Numeric](expr: Expr[A], exponent: Expr[Float]) extends Expr[A] {
  override def name: String = "Pow"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override val inputs: Seq[Expr[_]] = Seq(expr, exponent.as("exponent").cast[A])
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val koef = Pow(expr.cast[R], Minus(exponent, 1f.const))
      val local = Multiply(koef, exponent.cast[R])
      List(Multiply(local, parentGrad))
    }
  }
}

case class Sqrt[A: TensorType: Numeric](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Sqrt"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val local = Multiply(kernels.pow(expr.cast[R], -0.5f), 0.5f.const.cast[R])
      List(Multiply(local, parentGrad))
    }
  }
}

case class Exp[A: TensorType: Numeric](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Exp"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(Multiply(Exp(expr).cast[R], parentGrad))
    }
  }
}

case class Div[A: TensorType: Numeric](left: Expr[A], right: Expr[A]) extends Expr[A] {
  require(
    left.broadcastableAny(right),
    s"cannot divide tensors with shapes ${left.shape} / ${right.shape}")
  override def name: String = "Div"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val parentShape = parentGrad.shape
      val shrinkRightAxises = parentShape.broadcastableAxises(right.shape).toList
      val shrinkLeftAxises = parentShape.broadcastableAxises(left.shape).toList
      List(
        Sum(Div(parentGrad, right.cast[R]), shrinkLeftAxises).reshape(left.shape),
        Sum(
          Negate(Div(Multiply(left.cast[R], parentGrad), Multiply(right, right).cast[R])),
          shrinkRightAxises).reshape(right.shape))
    }
  }
}

case class Sum[A: TensorType: Numeric] private (expr: Expr[A], axises: Seq[Int]) extends Expr[A] {
  override def name: String = "Sum"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = expr.shape.remove(axises: _*)
  override val inputs: Seq[Expr[_]] =
    Seq(expr, Tensor.vector(axises.map(_.toLong): _*).const.as("axises"))
  override def compiler: core.Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      // we need to recover reduced axises with 1, cause broadcasting will not always work
      val parentShape = axises.foldLeft(parentGrad.shape)((s, axis) => s.insert(axis, 1))
      List(Multiply(kernels.ones[R](expr.shape), parentGrad.reshape(parentShape)))
    }
  }
}

object Sum {
  def apply[A: TensorType: Numeric](expr: Expr[A], axises: Seq[Int]): Expr[A] =
    if (expr.isScalar || axises.isEmpty) expr else new Sum(expr, axises)
}

case class Mean[A: TensorType: Numeric] private (expr: Expr[A], axises: Seq[Int]) extends Expr[A] {
  override def name: String = "Mean"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = expr.shape.remove(axises: _*)
  override val inputs: Seq[Expr[_]] =
    Seq(expr, Tensor.vector(axises.map(_.toLong): _*).const.as("axises"))
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      // we need to recover reduced axises with 1, cause broadcasting will not always work
      val parentShape = axises.foldLeft(parentGrad.shape)((s, axis) => s.insert(axis, 1))
      val size = expr.shape.select(axises: _*).power
      List(
        Div(
          Multiply(kernels.ones[R](expr.shape), parentGrad.reshape(parentShape)),
          size.const.cast[R]))
    }
  }
}

object Mean {
  def apply[A: TensorType: Numeric](expr: Expr[A], axises: Seq[Int]): Expr[A] =
    if (expr.isScalar || axises.isEmpty) expr else new Mean(expr, axises)
}

case class Max[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[A] {
  require(
    left.broadcastableAny(right),
    s"cannot compare tensors with shapes ${left.shape} <> ${right.shape}")
  override def name: String = "Maximum"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(
        Multiply((left > right).cast[R], parentGrad),
        Multiply((left < right).cast[R], parentGrad))
    }
  }
}

case class Min[A: TensorType](left: Expr[A], right: Expr[A]) extends Expr[A] {
  require(
    left.broadcastableAny(right),
    s"cannot compare tensors with shapes ${left.shape} <> ${right.shape}")
  override def name: String = "Minimum"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = left.shape max right.shape
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(
        Multiply((left < right).cast[R], parentGrad),
        Multiply((left > right).cast[R], parentGrad))
    }
  }
}

case class Abs[A: TensorType: Numeric](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Abs"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(Abs(parentGrad))
    }
  }
}

case class Round[A: TensorType: Numeric](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Rint"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
}

case class Transpose[A: TensorType: Numeric] private (expr: Expr[A], perm: Seq[Int])
    extends Expr[A] {
  override def name: String = "Transpose"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = expr.shape.permute(perm: _*)
  override val inputs: Seq[Expr[_]] =
    Seq(expr, Tensor.vector(perm.map(_.toLong): _*).const.as("perm"))
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(Transpose(parentGrad))
    }
  }
}

object Transpose {
  def apply[A: TensorType: Numeric](expr: Expr[A], perm: Seq[Int]): Expr[A] =
    if (expr.isScalar) expr else new Transpose(expr, perm)
  def apply[A: TensorType: Numeric](expr: Expr[A]): Expr[A] =
    apply(expr, (0 until expr.rank).reverse)
}

case class MatMul[A: TensorType: Numeric](left: Expr[A], right: Expr[A]) extends Expr[A] {
  require(
    left.rank == 2 && right.rank == 2,
    s"rank cannot be > 2 but got tensors with shapes ${left.shape} * ${right.shape}")
  require(
    left.shape.last == right.shape.head,
    s"cannot matmul tensors with shapes ${left.shape} * ${right.shape}")
  override def name: String = "MatMul"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = Shape(left.shape.head, right.shape.last)
  override def inputs: Seq[Expr[_]] = Seq(left, right)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(
        MatMul(parentGrad, Transpose(right).cast[R]),
        MatMul(Transpose(left).cast[R], parentGrad))
    }
  }
}

case class Sigmoid[A: TensorType: Numeric: Floating](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Sigmoid"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val s = Sigmoid(expr)
      val grad = Multiply(s, Minus(1.0f.const.cast[A], s))
      List(Multiply(grad.cast[R], parentGrad))
    }
  }
}

case class Tanh[A: TensorType: Numeric: Floating](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Tanh"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val grad = Minus(1f.const.cast[A], Pow(Tanh(expr), 2.0f.const))
      List(Multiply(grad.cast[R], parentGrad))
    }
  }
}

case class Log[A: TensorType: Numeric](expr: Expr[A]) extends Expr[A] {
  override def name: String = "Log"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Numeric: Floating: TensorType](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      List(Div(parentGrad, expr.cast[R]))
    }
  }
}

trait Kernels {

  def zeros[A: TensorType: Numeric](shape: Int*): Expr[A] = fill(shape: _*)(Numeric[A].zero)

  def zeros[A: TensorType: Numeric](shape: Shape): Expr[A] = fill(shape)(Numeric[A].zero)

  def ones[A: TensorType: Numeric](shape: Int*): Expr[A] = fill(shape: _*)(Numeric[A].one)

  def ones[A: TensorType: Numeric](shape: Shape): Expr[A] = fill(shape)(Numeric[A].one)

  def plus[A: TensorType: Numeric, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[A] = Plus(left, c.convert(right))

  def plus[A: TensorType: Numeric](first: Expr[A], second: Expr[A], third: Expr[A], rest: Expr[A]*): Expr[A] =
    PlusN(first +: second +: third +: rest.toList)

  def plus[A: TensorType: Numeric](expr: Seq[Expr[A]]): Expr[A] = PlusN(expr.toList)

  def minus[A: TensorType: Numeric, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[A] = Minus(left, c.convert(right))

  def negate[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = Negate(expr)

  def multiply[A: TensorType: Numeric, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[A] =
    Multiply(left, c.convert(right))

  def div[A: TensorType: Numeric, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[A] = Div(left, c.convert(right))

  def pow[A: TensorType: Numeric](expr: Expr[A], exponent: Expr[Float]): Expr[A] =
    Pow(expr, exponent)

  def pow[A: TensorType: Numeric](expr: Expr[A], exponent: Float): Expr[A] =
    Pow(expr, Tensor.scalar(exponent).const)

  def sqrt[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = Sqrt(expr)

  def sqrtZeroSafe[A: TensorType: Numeric](out: Expr[A], epsilon: Expr[A]): Expr[A] =
    sqrt(plus(out, epsilon))

  def exp[A: TensorType: Numeric: Floating](expr: Expr[A]): Expr[A] = Exp(expr)

  def sum[A: TensorType: Numeric](out: Expr[A], axises: Seq[Int]): Expr[A] = Sum(out, axises)
  def sum[A: TensorType: Numeric](out: Expr[A]): Expr[A] = sum(out, 0 until out.rank)

  def mean[A: TensorType: Numeric](expr: Expr[A], axises: Seq[Int]): Expr[A] = Mean(expr, axises)
  def mean[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = mean(expr, 0 until expr.rank)

  def max[A: TensorType, C](left: Expr[A], right: C)(implicit c: Convertible[C, Expr[A]]): Expr[A] =
    Max(left, c.convert(right))

  def min[A: TensorType, C](left: Expr[A], right: C)(implicit c: Convertible[C, Expr[A]]): Expr[A] =
    Min(left, c.convert(right))

  def abs[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = Abs(expr)

  def round[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = Round(expr)

  def transpose[A: TensorType: Numeric](expr: Expr[A], perm: Seq[Int]): Expr[A] =
    Transpose(expr, perm)

  def transpose[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = Transpose(expr)

  def matmul[A: TensorType: Numeric, C](left: Expr[A], right: C)(
      implicit c: Convertible[C, Expr[A]]): Expr[A] = MatMul(left, c.convert(right))

  def decayingAvg[A: TensorType: Numeric](avg: Expr[A], next: Expr[A], decay: Expr[A]): Expr[A] = {
    // todo: try to rewrite with ops
    plus(multiply(decay, avg), multiply(minus(1.0f.const.cast[A], decay), next))
  }

  def boost[A: TensorType: Numeric](out: Expr[A], rate: Expr[A], iter: Expr[Int]): Expr[A] = {
    // todo: try to rewrite with ops
    div(out, minus(Numeric[A].one.const, pow(rate, iter.cast[Float])))
  }

  def sigmoid[A: TensorType: Numeric: Floating](expr: Expr[A]): Expr[A] = Sigmoid(expr)

  def tanh[A: TensorType: Numeric: Floating](expr: Expr[A]): Expr[A] = Tanh(expr)

  def log[A: TensorType: Numeric](expr: Expr[A]): Expr[A] = Log(expr)
}

object kernels extends Kernels {

  case class NumericOps[A: TensorType: Numeric](expr: Expr[A]) {
    import org.scanet.math.alg.{kernels => f}

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
      * @param right side
      * @return a result of addition
      */
    def plus(right: Expr[A]): Expr[A] = f.plus(expr, right)
    def +(right: Expr[A]): Expr[A] = f.plus(expr, right)

    /** Subtract two tensors. Supports broadcasting.
      *
      * {{{(2.const - Tensor.vector(5, 10).const).eval should be(Tensor.vector(-3, -8))}}}
      *
      * @param right side
      * @return a result of subtraction
      */
    def minus(right: Expr[A]): Expr[A] = f.minus(expr, right)
    def -(right: Expr[A]): Expr[A] = f.minus(expr, right)

    /** Negate a tensor.
      *
      * {{{(Tensor.vector(1, 2, 3).const.negate.eval should be(Tensor.vector(-1, -2, -3))}}}
      *
      * @return a result of negation
      */
    def negate: Expr[A] = f.negate(expr)
    def unary_- : Expr[A] = f.negate(expr)

    /** Element-wise division. Supports broadcasting.
      *
      * {{{(Tensor.vector(5, 10, 15).const /:/ 5.const).eval should be(Tensor.vector(1, 2, 3))}}}
      *
      * @param right side
      * @return a result of division
      */
    def div(right: Expr[A]): Expr[A] = f.div(expr, right)
    def /(right: Expr[A]): Expr[A] = f.div(expr, right)

    /** Element-wise multiplication. Supports broadcasting.
      *
      * {{{Tensor.vector(1, 2, 3).const *:* 5.const).eval should be(Tensor.vector(5, 10, 15))}}}
      *
      * @param right side
      * @return a result of multiplication
      */
    def multiply(right: Expr[A]): Expr[A] =
      f.multiply(expr, right)
    def *(right: Expr[A]): Expr[A] = f.multiply(expr, right)

    /** Raises the tensor to the power of the exponent
      *
      * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
      *
      * @return tensor `^` exponent
      */
    def pow(exponent: Float): Expr[A] = f.pow(expr, exponent.const)

    /** Raises the tensor to the power of the exponent
      *
      * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.pow(2).eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
      *
      * @return tensor `^` exponent
      */
    def pow(exponent: Expr[Float]): Expr[A] = f.pow(expr, exponent)

    /** Raises the tensor to the power of two
      *
      * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.sqr.eval should be(Tensor.vector(1.0f, 4.0f, 9.0f))}}}
      *
      * @return tensor `^` 2
      */
    def sqr: Expr[A] = pow(2.0f)

    /** Returns square root of the given tensor
      *
      * {{{Tensor.vector(1.0f, 4.0f, 9.0f).const.sqrt.eval should be(Tensor.vector(1.0f, 2.0f, 3.0f))}}}
      *
      * @return  tensor `^` 0.5
      */
    def sqrt: Expr[A] = f.sqrt(expr)

    /** Returns square root of the given tensor which never reaches zero
      * It is equivalent to `(out + epsilon) ^ 0.5`.
      *
      * {{{8f.const.sqrtZeroSafe(1f.const).eval should be(Tensor.scalar(3f))}}}
      *
      * @return tensor
      */
    def sqrtZeroSafe(epsilon: Expr[A]): Expr[A] = f.sqrtZeroSafe(expr, epsilon)

    /** Computes the sum of elements across dimensions of a tensor.
      *
      * Reduces `tensor` along the dimensions given in `axises`.
      * The rank of the tensor is reduced by 1 for each entry in `axises`.
      *
      * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum(Seq(0)).eval should be(Tensor.vector(5, 7, 9))}}}
      *
      * @param axises to sum
      * @return tensor with summed values
      */
    def sum(axises: Seq[Int]): Expr[A] = f.sum(expr, axises)

    /** Computes the sum of elements across all dimensions of a tensor.
      *
      * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.sum.eval should be(Tensor.scalar(21))}}}
      *
      * @return tensor with summed values
      */
    def sum: Expr[A] = f.sum(expr)

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
      * @param right side
      * @return a result of multiplication
      */
    def matmul(right: Expr[A]): Expr[A] = f.matmul(expr, right)

    /** Computes the mean of elements across dimensions of a tensor.
      *
      * Reduces `out` along the dimensions given in `axises`.
      * The rank of the tensor is reduced by 1 for each entry in `axises`.
      *
      * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.mean(Seq(0)).eval should be(Tensor.vector(5, 7, 9))}}}
      *
      * @param axises to sum
      * @return tensor with mean values
      */
    def mean(axises: Seq[Int]): Expr[A] = f.mean(expr, axises)

    /** Computes the mean of elements across all dimensions of a tensor.
      *
      * {{{Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const.mean.eval should be(Tensor.scalar(21))}}}
      *
      * @return tensor with mean values
      */
    def mean: Expr[A] = f.mean(expr)

    /** Shuffle dimensions of `out` according to a permutation.
      *
      * {{{
      * val before = Tensor(range(1, 13), Shape(2, 2, 3))
      * val after = Tensor(Array(1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12), Shape(2, 3, 2))
      * before.const.transpose(Seq(0, 2, 1)).eval should be(after)
      * }}}
      *
      * @param perm dimensions permutations
      * @return transposed tensor
      */
    def transpose(perm: Seq[Int]): Expr[A] = f.transpose(expr, perm)

    /** Transpose given tensor - flip dimensions over its diagonal.
      *
      * {{{
      * val matrix = Tensor.matrix(Array(1, 2), Array(3, 4)).const
      * matrix.transpose.eval should be(Tensor.matrix(Array(1, 3), Array(2, 4)))
      * }}}
      *
      * @return transposed tensor
      */
    def transpose: Expr[A] = f.transpose(expr)

    /** Compute decaying average of given value by putting more weight into latest (`next`) value.
      * It is equivalent to `(prev * decay) + ((1 - decay) * next)`.
      *
      * {{{10.0.const.decayingAvg(5.0.const, 0.9.const).eval should be(Tensor.scalar(9.5))}}}
      *
      * @param next current value
      * @param decay rate of decay
      * @return decaying average
      */
    def decayingAvg(next: Expr[A], decay: Expr[A]): Expr[A] = f.decayingAvg(expr, next, decay)

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
      * @param rate boost rate, should be < 1
      * @param iter computation iteration
      * @return unbiased value
      */
    def boost(rate: Expr[A], iter: Expr[Int]): Expr[A] = f.boost(expr, rate, iter)

    /** Returns tensor with absolute values
      *
      * {{{Tensor.vector(-1, 2, -3).abs.eval should be(Tensor.vector(1, 2, 3)}}}
      *
      * @return |tensor|
      */
    def abs: Expr[A] = f.abs(expr)

    /** Computes natural logarithm of `x` element-wise
      *
      * @return natural logarithm
      */
    def log: Expr[A] = f.log(expr)

    /** Computes element-wise integer closest to `x`.
      *
      * @return rounded tensor
      */
    def round: Expr[A] = f.round(expr)
  }

  case class FloatingOps[A: TensorType: Numeric: Floating](expr: Expr[A]) {
    import org.scanet.math.alg.{kernels => f}

    /** Computes exponential of a given tensor element wise.
      *
      * {x.exp} is equal to {e.pow(x)}, however, exponential function is preferred cause it has
      * optimized kernel and identity derivative.
      *
      * `e` denotes Euler's number and is approximately equal to `2.718281`.
      *
      * {{{Tensor.vector(1.0f, 2.0f, 3.0f).const.exp.eval should be(Tensor.vector(2.7182817f, 7.389056f, 20.085537f))}}}
      *
      * @return  exponent `^` tensor
      */
    def exp: Expr[A] = f.exp(expr)

    /** Computes sigmoid of `x` element-wise. Specifically
      *
      * {{{y = 1 / (1 + e^(-x))}}}
      *
      * @return sigmoid
      */
    def sigmoid: Expr[A] = f.sigmoid(expr)

    /** Computes hyperbolic tangent of `x` element-wise.
      *
      * Given an input tensor, this function computes hyperbolic tangent of every
      * element in the tensor. Input range is `[-inf, inf]` and output range is `[-1,1]`.
      *
      * {{{Tensor.scalar(0.5f).const.tanh.eval should be(Tensor.scalar(0.46211717f))}}}
      *
      * @return tanh
      */
    def tanh: Expr[A] = f.tanh(expr)
  }

  trait Syntax extends Kernels {
    implicit def toMathKernelNumericOps[A: TensorType: Numeric](expr: Expr[A]): NumericOps[A] =
      new NumericOps[A](expr)
    implicit def toMathKernelFloatingOps[A: TensorType: Numeric: Floating](
        expr: Expr[A]): FloatingOps[A] =
      new FloatingOps[A](expr)
  }

  object syntax extends Syntax
}
