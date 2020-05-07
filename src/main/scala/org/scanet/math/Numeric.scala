package org.scanet.math

import org.scanet.core.{TensorTypeByte, TensorTypeDouble, TensorTypeFloat, TensorTypeInt, TensorTypeLong}
import simulacrum.{op, typeclass}

@typeclass trait Eq[A] {
  @op("===", alias = true)
  def eqv(x: A, y: A): Boolean
  @op("=!=", alias = true)
  def neqv(x: A, y: A): Boolean = !eqv(x, y)
}

@typeclass trait Order[A] extends Eq[A] {
  def compare(x: A, y: A): Int
  @op(">", alias = true)
  def gt(x: A, y: A): Boolean = compare(x, y) > 0
  @op(">=", alias = true)
  def gte(x: A, y: A): Boolean = compare(x, y) >= 0
  @op("<", alias = true)
  def lt(x: A, y: A): Boolean = compare(x, y) < 0
  @op("<=", alias = true)
  def lte(x: A, y: A): Boolean = compare(x, y) <= 0
  override def eqv(x: A, y: A): Boolean = compare(x, y) == 0
}

@typeclass trait Semiring[A] {

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
   * @tparam B type which can be converted into output
   * @return a result of addition
   */
  // todo: figure out why + operator is not resolved
  @op("+", alias = true)
  def plus[B](left: A, right: B)(implicit c: Convertible[B, A]): A

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
   * @tparam B type which can be converted into output
   * @return a result of multiplication
   */
  @op("*", alias = true)
  def multiply[B](left: A, right: B)(implicit c: Convertible[B, A]): A
}

@typeclass trait Rng[A] extends Semiring[A] {
  def zero: A
  @op("-", alias = true)
  def minus[B](left: A, right: B)(implicit c: Convertible[B, A]): A
  // todo: figure out why unary_- operator is not resolved
  @op("unary_-", alias = true)
  def negate(a: A): A
}

@typeclass trait Rig[A] extends Semiring[A] {
  def one: A
}

@typeclass trait Ring[A] extends Rng[A] with Rig[A] {}

@typeclass trait Field[A] extends Ring[A] {
  @op("/", alias = true)
  def div[B](left: A, right: B)(implicit c: Convertible[B, A]): A
}

@typeclass trait Numeric[A] extends Field[A] with Order[A] {}

object Numeric {

  trait Instances {

    implicit def floatInst: Numeric[Float] = new NumericFloat {}
    implicit def doubleInst: Numeric[Double] = new NumericDouble {}
    implicit def longInst: Numeric[Long] = new NumericLong {}
    implicit def intInst: Numeric[Int] = new NumericInt {}
    implicit def byteInst: Numeric[Byte] = new NumericByte {}

    implicit def identityConvertible[A]: Convertible[A, A] = new IdentityConvertible[A] {}

    implicit def fromFloatToFloat: Convertible[Float, Float] = new FromFloatToFloat {}
    implicit def fromFloatToDouble: Convertible[Float, Double] = new FromFloatToDouble {}
    implicit def fromFloatToLong: Convertible[Float, Long] = new FromFloatToLong {}
    implicit def fromFloatToInt: Convertible[Float, Int] = new FromFloatToInt {}
    implicit def fromFloatToByte: Convertible[Float, Byte] = new FromFloatToByte {}

    implicit def fromDoubleToFloat: Convertible[Double, Float] = new FromDoubleToFloat {}
    implicit def fromDoubleToDouble: Convertible[Double, Double] = new FromDoubleToDouble {}
    implicit def fromDoubleToLong: Convertible[Double, Long] = new FromDoubleToLong {}
    implicit def fromDoubleToInt: Convertible[Double, Int] = new FromDoubleToInt {}
    implicit def fromDoubleToByte: Convertible[Double, Byte] = new FromDoubleToByte {}

    implicit def fromLongToFloat: Convertible[Long, Float] = new FromLongToFloat {}
    implicit def fromLongToDouble: Convertible[Long, Double] = new FromLongToDouble {}
    implicit def fromLongToLong: Convertible[Long, Long] = new FromLongToLong {}
    implicit def fromLongToInt: Convertible[Long, Int] = new FromLongToInt {}
    implicit def fromLongToByte: Convertible[Long, Byte] = new FromLongToByte {}

    implicit def fromIntToFloat: Convertible[Int, Float] = new FromIntToFloat {}
    implicit def fromIntToDouble: Convertible[Int, Double] = new FromIntToDouble {}
    implicit def fromIntToLong: Convertible[Int, Long] = new FromIntToLong {}
    implicit def fromIntToInt: Convertible[Int, Int] = new FromIntToInt {}
    implicit def fromIntToByte: Convertible[Int, Byte] = new FromIntToByte {}

    implicit def fromByteToFloat: Convertible[Byte, Float] = new FromByteToFloat {}
    implicit def fromByteToDouble: Convertible[Byte, Double] = new FromByteToDouble {}
    implicit def fromByteToLong: Convertible[Byte, Long] = new FromByteToLong {}
    implicit def fromByteToInt: Convertible[Byte, Int] = new FromByteToInt {}
    implicit def fromByteToByte: Convertible[Byte, Byte] = new FromByteToByte {}
  }

  trait Syntax extends Instances with Semiring.ToSemiringOps
    with Rng.ToRngOps with Rig.ToRigOps with Ring.ToRingOps
    with Field.ToFieldOps with Eq.ToEqOps with Order.ToOrderOps
    with Numeric.ToNumericOps

  object syntax extends Syntax
}

trait NumericFloat extends Numeric[Float] with TensorTypeFloat {
  override def one: Float = 1.0f
  override def zero: Float = 0.0f
  override def plus[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left + Convertible[B, Float].convert(right)
  override def multiply[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left * Convertible[B, Float].convert(right)
  override def minus[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left - Convertible[B, Float].convert(right)
  override def negate(a: Float): Float = -a
  override def div[B](left: Float, right: B)(implicit c: Convertible[B, Float]): Float =
    left / Convertible[B, Float].convert(right)
  override def compare(x: Float, y: Float): Int = x.compareTo(y)
}

trait NumericDouble extends Numeric[Double] with TensorTypeDouble {
  override def one: Double = 1.0d
  override def zero: Double = 0.0d
  override def plus[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left + Convertible[B, Double].convert(right)
  override def multiply[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left * Convertible[B, Double].convert(right)
  override def minus[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left - Convertible[B, Double].convert(right)
  override def negate(a: Double): Double = -a
  override def div[B](left: Double, right: B)(implicit c: Convertible[B, Double]): Double =
    left / Convertible[B, Double].convert(right)
  override def compare(x: Double, y: Double): Int = x.compareTo(y)
}

trait NumericLong extends Numeric[Long] with TensorTypeLong {
  override def one: Long = 1L
  override def zero: Long = 0L
  override def plus[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left + Convertible[B, Long].convert(right)
  override def multiply[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left * Convertible[B, Long].convert(right)
  override def minus[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left - Convertible[B, Long].convert(right)
  override def negate(a: Long): Long = -a
  override def div[B](left: Long, right: B)(implicit c: Convertible[B, Long]): Long =
    left / Convertible[B, Long].convert(right)
  override def compare(x: Long, y: Long): Int = x.compareTo(y)
}

trait NumericInt extends Numeric[Int] with TensorTypeInt {
  override def one: Int = 1
  override def zero: Int = 0
  override def plus[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left + Convertible[B, Int].convert(right)
  override def multiply[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left * Convertible[B, Int].convert(right)
  override def minus[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left - Convertible[B, Int].convert(right)
  override def negate(a: Int): Int = -a
  override def div[B](left: Int, right: B)(implicit c: Convertible[B, Int]): Int =
    left / Convertible[B, Int].convert(right)
  override def compare(x: Int, y: Int): Int = x.compareTo(y)
}

trait NumericByte extends Numeric[Byte] with TensorTypeByte {
  override def one: Byte = 1
  override def zero: Byte = 0
  override def plus[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left + Convertible[B, Byte].convert(right)).toByte
  override def multiply[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left * Convertible[B, Byte].convert(right)).toByte
  override def minus[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left - Convertible[B, Byte].convert(right)).toByte
  override def negate(a: Byte): Byte = (-a).toByte
  override def div[B](left: Byte, right: B)(implicit c: Convertible[B, Byte]): Byte =
    (left / Convertible[B, Byte].convert(right)).toByte
  override def compare(x: Byte, y: Byte): Int = x.compareTo(y)
}

trait IdentityConvertible[A] extends Convertible[A, A] {
  override def convert(a: A): A = a
}

trait FromFloatToFloat extends Convertible[Float, Float] {
  override def convert(a: Float): Float = a
}
trait FromFloatToDouble extends Convertible[Float, Double] {
  override def convert(a: Float): Double = a.toDouble
}
trait FromFloatToLong extends Convertible[Float, Long] {
  override def convert(a: Float): Long = a.toLong
}
trait FromFloatToInt extends Convertible[Float, Int] {
  override def convert(a: Float): Int = a.toInt
}
trait FromFloatToByte extends Convertible[Float, Byte] {
  override def convert(a: Float): Byte = a.toByte
}

trait FromDoubleToFloat extends Convertible[Double, Float] {
  override def convert(a: Double): Float = a.toFloat
}
trait FromDoubleToDouble extends Convertible[Double, Double] {
  override def convert(a: Double): Double = a
}
trait FromDoubleToLong extends Convertible[Double, Long] {
  override def convert(a: Double): Long = a.toLong
}
trait FromDoubleToInt extends Convertible[Double, Int] {
  override def convert(a: Double): Int = a.toInt
}
trait FromDoubleToByte extends Convertible[Double, Byte] {
  override def convert(a: Double): Byte = a.toByte
}

trait FromLongToFloat extends Convertible[Long, Float] {
  override def convert(a: Long): Float = a.toFloat
}
trait FromLongToDouble extends Convertible[Long, Double] {
  override def convert(a: Long): Double = a.toDouble
}
trait FromLongToLong extends Convertible[Long, Long] {
  override def convert(a: Long): Long = a
}
trait FromLongToInt extends Convertible[Long, Int] {
  override def convert(a: Long): Int = a.toInt
}
trait FromLongToByte extends Convertible[Long, Byte] {
  override def convert(a: Long): Byte = a.toByte
}

trait FromIntToFloat extends Convertible[Int, Float] {
  override def convert(a: Int): Float = a.toFloat
}
trait FromIntToDouble extends Convertible[Int, Double] {
  override def convert(a: Int): Double = a.toDouble
}
trait FromIntToLong extends Convertible[Int, Long] {
  override def convert(a: Int): Long = a.toLong
}
trait FromIntToInt extends Convertible[Int, Int] {
  override def convert(a: Int): Int = a.toInt
}
trait FromIntToByte extends Convertible[Int, Byte] {
  override def convert(a: Int): Byte = a.toByte
}

trait FromByteToFloat extends Convertible[Byte, Float] {
  override def convert(a: Byte): Float = a.toByte
}
trait FromByteToDouble extends Convertible[Byte, Double] {
  override def convert(a: Byte): Double = a.toDouble
}
trait FromByteToLong extends Convertible[Byte, Long] {
  override def convert(a: Byte): Long = a.toLong
}
trait FromByteToInt extends Convertible[Byte, Int] {
  override def convert(a: Byte): Int = a.toInt
}
trait FromByteToByte extends Convertible[Byte, Byte] {
  override def convert(a: Byte): Byte = a
}