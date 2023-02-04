package scanet.core

import scanet.core.TensorType._

trait Convertible[A, B] extends Serializable {
  def convert(a: A): B
}

object Convertible {

  def apply[A, B](implicit convertible: Convertible[A, B]): Convertible[A, B] = convertible

  implicit def identityConvertible[A]: Convertible[A, A] = new IdentityConvertible[A] {}

  trait Instances {

    implicit val fromFloatToFloat: Convertible[Float, Float] = new FromFloatToFloat {}
    implicit val fromFloatToDouble: Convertible[Float, Double] = new FromFloatToDouble {}
    implicit val fromFloatToLong: Convertible[Float, Long] = new FromFloatToLong {}
    implicit val fromFloatToInt: Convertible[Float, Int] = new FromFloatToInt {}
    implicit val fromFloatToByte: Convertible[Float, Byte] = new FromFloatToByte {}

    implicit val fromDoubleToFloat: Convertible[Double, Float] = new FromDoubleToFloat {}
    implicit val fromDoubleToDouble: Convertible[Double, Double] = new FromDoubleToDouble {}
    implicit val fromDoubleToLong: Convertible[Double, Long] = new FromDoubleToLong {}
    implicit val fromDoubleToInt: Convertible[Double, Int] = new FromDoubleToInt {}
    implicit val fromDoubleToByte: Convertible[Double, Byte] = new FromDoubleToByte {}

    implicit val fromLongToFloat: Convertible[Long, Float] = new FromLongToFloat {}
    implicit val fromLongToDouble: Convertible[Long, Double] = new FromLongToDouble {}
    implicit val fromLongToLong: Convertible[Long, Long] = new FromLongToLong {}
    implicit val fromLongToInt: Convertible[Long, Int] = new FromLongToInt {}
    implicit val fromLongToByte: Convertible[Long, Byte] = new FromLongToByte {}

    implicit val fromIntToFloat: Convertible[Int, Float] = new FromIntToFloat {}
    implicit val fromIntToDouble: Convertible[Int, Double] = new FromIntToDouble {}
    implicit val fromIntToLong: Convertible[Int, Long] = new FromIntToLong {}
    implicit val fromIntToInt: Convertible[Int, Int] = new FromIntToInt {}
    implicit val fromIntToByte: Convertible[Int, Byte] = new FromIntToByte {}

    implicit val fromByteToFloat: Convertible[Byte, Float] = new FromByteToFloat {}
    implicit val fromByteToDouble: Convertible[Byte, Double] = new FromByteToDouble {}
    implicit val fromByteToLong: Convertible[Byte, Long] = new FromByteToLong {}
    implicit val fromByteToInt: Convertible[Byte, Int] = new FromByteToInt {}
    implicit val fromByteToByte: Convertible[Byte, Byte] = new FromByteToByte {}

    // todo: rewrite all
    implicit def fromNumericToFloat[A: Numeric]: Convertible[A, Float] = {
      case v: Float  => v
      case v: Double => v.toFloat
      case v: Long   => v.toFloat
      case v: Int    => v.toFloat
      case v: Byte   => v.toFloat
      case other     => error(s"Value $other cannot be converted to Float")
    }
  }

  trait AllSyntax extends Instances

  object syntax extends AllSyntax
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
