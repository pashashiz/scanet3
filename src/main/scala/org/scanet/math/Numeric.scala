package org.scanet.math

import org.scanet.core.{TfType, TfTypeByte, TfTypeDouble, TfTypeFloat, TfTypeInt, TfTypeLong}
import org.tensorflow.DataType
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
  // todo: figure out why + operator is not resolved
  @op("+", alias = true)
  def plus[B: ConvertableFrom](left: A, right: B): A
  @op("*", alias = true)
  def multiply[B: ConvertableFrom](left: A, right: B): A
}

@typeclass trait Rng[A] extends Semiring[A] {
  def zero: A
  @op("-", alias = true)
  def minus[B: ConvertableFrom](left: A, right: B): A
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
  def div[B: ConvertableFrom](left: A, right: B): A
}

@typeclass trait ConvertableTo[A] {
  def fromByte(n: Byte): A
  def fromInt(n: Int): A
  def fromLong(n: Long): A
  def fromFloat(n: Float): A
  def fromDouble(n: Double): A
  def fromType[B: ConvertableFrom](b: B): A
}

@typeclass trait ConvertableFrom[A] {
  def toByte(a: A): Byte
  def toInt(a: A): Int
  def toLong(a: A): Long
  def toFloat(a: A): Float
  def toDouble(a: A): Double
  def toType[B: ConvertableTo](a: A): B
  def asString(a: A): String
}

@typeclass trait Numeric[A] extends TfType[A] with Field[A] with Order[A] with ConvertableTo[A] with ConvertableFrom[A] {}

object Numeric {

  trait Instances {
    implicit def floatInst: Numeric[Float] = new NumericFloat {}
    implicit def doubleInst: Numeric[Double] = new NumericDouble {}
    implicit def longInst: Numeric[Long] = new NumericLong {}
    implicit def intInst: Numeric[Int] = new NumericInt {}
    implicit def byteInst: Numeric[Byte] = new NumericByte {}
  }

  trait Syntax extends Instances with Semiring.ToSemiringOps
    with Rng.ToRngOps with Rig.ToRigOps with Ring.ToRingOps
    with Field.ToFieldOps with Eq.ToEqOps with Order.ToOrderOps
    with Numeric.ToNumericOps

  object syntax extends Syntax
}

trait NumericFloat extends Numeric[Float] with TfTypeFloat with ConvertableFromFloat with ConvertableToFloat {
  override def one: Float = 1.0f
  override def zero: Float = 0.0f
  override def plus[B: ConvertableFrom](left: Float, right: B): Float =
    left + ConvertableFrom[B].toFloat(right)
  override def multiply[B: ConvertableFrom](left: Float, right: B): Float =
    left * ConvertableFrom[B].toFloat(right)
  override def minus[B: ConvertableFrom](left: Float, right: B): Float =
    left - ConvertableFrom[B].toFloat(right)
  override def negate(a: Float): Float = -a
  override def div[B: ConvertableFrom](left: Float, right: B): Float =
    left / ConvertableFrom[B].toFloat(right)
  override def compare(x: Float, y: Float): Int = x.compareTo(y)
}

trait NumericDouble extends Numeric[Double] with TfTypeDouble with ConvertableFromDouble with ConvertableToDouble {
  override def one: Double = 1.0d
  override def zero: Double = 0.0d
  override def plus[B: ConvertableFrom](left: Double, right: B): Double =
    left + ConvertableFrom[B].toFloat(right)
  override def multiply[B: ConvertableFrom](left: Double, right: B): Double =
    left * ConvertableFrom[B].toFloat(right)
  override def minus[B: ConvertableFrom](left: Double, right: B): Double =
    left - ConvertableFrom[B].toFloat(right)
  override def negate(a: Double): Double = -a
  override def div[B: ConvertableFrom](left: Double, right: B): Double =
    left / ConvertableFrom[B].toDouble(right)
  override def compare(x: Double, y: Double): Int = x.compareTo(y)
}

trait NumericLong extends Numeric[Long] with TfTypeLong with ConvertableFromLong with ConvertableToLong {
  override def one: Long = 1L
  override def zero: Long = 0L
  override def plus[B: ConvertableFrom](left: Long, right: B): Long =
    left + ConvertableFrom[B].toLong(right)
  override def multiply[B: ConvertableFrom](left: Long, right: B): Long =
    left * ConvertableFrom[B].toLong(right)
  override def minus[B: ConvertableFrom](left: Long, right: B): Long =
    left - ConvertableFrom[B].toLong(right)
  override def negate(a: Long): Long = -a
  override def div[B: ConvertableFrom](left: Long, right: B): Long =
    left / ConvertableFrom[B].toLong(right)
  override def compare(x: Long, y: Long): Int = x.compareTo(y)
}

trait NumericInt extends Numeric[Int] with TfTypeInt with ConvertableFromInt with ConvertableToInt {
  override def one: Int = 1
  override def zero: Int = 0
  override def plus[B: ConvertableFrom](left: Int, right: B): Int =
    left + ConvertableFrom[B].toInt(right)
  override def multiply[B: ConvertableFrom](left: Int, right: B): Int =
    left * ConvertableFrom[B].toInt(right)
  override def minus[B: ConvertableFrom](left: Int, right: B): Int =
    left - ConvertableFrom[B].toInt(right)
  override def negate(a: Int): Int = -a
  override def div[B: ConvertableFrom](left: Int, right: B): Int =
    left / ConvertableFrom[B].toInt(right)
  override def compare(x: Int, y: Int): Int = x.compareTo(y)
}

trait NumericByte extends Numeric[Byte] with TfTypeByte with ConvertableFromByte with ConvertableToByte {
  override def one: Byte = 1
  override def zero: Byte = 0
  override def plus[B: ConvertableFrom](left: Byte, right: B): Byte =
    (left + ConvertableFrom[B].toByte(right)).toByte
  override def multiply[B: ConvertableFrom](left: Byte, right: B): Byte =
    (left * ConvertableFrom[B].toByte(right)).toByte
  override def minus[B: ConvertableFrom](left: Byte, right: B): Byte =
    (left - ConvertableFrom[B].toByte(right)).toByte
  override def negate(a: Byte): Byte = (-a).toByte
  override def div[B: ConvertableFrom](left: Byte, right: B): Byte =
    (left / ConvertableFrom[B].toByte(right)).toByte
  override def compare(x: Byte, y: Byte): Int = x.compareTo(y)
}