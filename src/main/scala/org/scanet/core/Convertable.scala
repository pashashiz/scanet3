package org.scanet.core

object Convertable {

  trait Instances {
    implicit def convertableToFloat: ConvertableTo[Float] = new ConvertableToFloat() {}
    implicit def convertableToDouble: ConvertableTo[Double] = new ConvertableToDouble() {}
    implicit def convertableToLong: ConvertableTo[Long] = new ConvertableToLong() {}
    implicit def convertableToInt: ConvertableTo[Int] = new ConvertableToInt() {}
    implicit def convertableToByte: ConvertableTo[Byte] = new ConvertableToByte() {}
    implicit def convertableFromFloat: ConvertableFrom[Float] = new ConvertableFromFloat() {}
    implicit def convertableFromDouble: ConvertableFrom[Double] = new ConvertableFromDouble() {}
    implicit def convertableFromLong: ConvertableFrom[Long] = new ConvertableFromLong() {}
    implicit def convertableFromInt: ConvertableFrom[Int] = new ConvertableFromInt() {}
    implicit def convertableFromByte: ConvertableFrom[Byte] = new ConvertableFromByte() {}
  }

  trait Syntax extends Instances with ConvertableTo.ToConvertableToOps with ConvertableFrom.ToConvertableFromOps

  object syntax extends Syntax
}

trait ConvertableToFloat extends ConvertableTo[Float] {
  override def fromByte(n: Byte): Float = n.toFloat
  override def fromInt(n: Int): Float = n.toFloat
  override def fromLong(n: Long): Float = n.toFloat
  override def fromFloat(n: Float): Float = n
  override def fromDouble(n: Double): Float = n.toFloat
  override def fromType[B: ConvertableFrom](b: B): Float = ConvertableFrom[B].toFloat(b)
}

trait ConvertableToDouble extends ConvertableTo[Double] {
  override def fromByte(n: Byte): Double = n.toDouble
  override def fromInt(n: Int): Double = n.toDouble
  override def fromLong(n: Long): Double = n.toDouble
  override def fromFloat(n: Float): Double = n.toDouble
  override def fromDouble(n: Double): Double = n
  override def fromType[B: ConvertableFrom](b: B): Double = ConvertableFrom[B].toDouble(b)
}

trait ConvertableToLong extends ConvertableTo[Long] {
  override def fromByte(n: Byte): Long = n.toLong
  override def fromInt(n: Int): Long = n.toLong
  override def fromLong(n: Long): Long = n
  override def fromFloat(n: Float): Long = n.toLong
  override def fromDouble(n: Double): Long = n.toLong
  override def fromType[B: ConvertableFrom](b: B): Long = ConvertableFrom[B].toLong(b)
}

trait ConvertableToInt extends ConvertableTo[Int] {
  override def fromByte(n: Byte): Int = n.toInt
  override def fromInt(n: Int): Int = n
  override def fromLong(n: Long): Int = n.toInt
  override def fromFloat(n: Float): Int = n.toInt
  override def fromDouble(n: Double): Int = n.toInt
  override def fromType[B: ConvertableFrom](b: B): Int = ConvertableFrom[B].toInt(b)
}

trait ConvertableToByte extends ConvertableTo[Byte] {
  override def fromByte(n: Byte): Byte = n
  override def fromInt(n: Int): Byte = n.toByte
  override def fromLong(n: Long): Byte = n.toByte
  override def fromFloat(n: Float): Byte = n.toByte
  override def fromDouble(n: Double): Byte = n.toByte
  override def fromType[B: ConvertableFrom](b: B): Byte = ConvertableFrom[B].toByte(b)
}

trait ConvertableFromFloat extends ConvertableFrom[Float] {
  override def toByte(a: Float): Byte = a.toByte
  override def toInt(a: Float): Int = a.toInt
  override def toLong(a: Float): Long = a.toLong
  override def toFloat(a: Float): Float = a
  override def toDouble(a: Float): Double = a.toDouble
  override def toType[B: ConvertableTo](a: Float): B = ConvertableTo[B].fromFloat(a)
  override def asString(a: Float): String = a.toString
}

trait ConvertableFromDouble extends ConvertableFrom[Double] {
  override def toByte(a: Double): Byte = a.toByte
  override def toInt(a: Double): Int = a.toInt
  override def toLong(a: Double): Long = a.toLong
  override def toFloat(a: Double): Float = a.toFloat
  override def toDouble(a: Double): Double = a
  override def toType[B: ConvertableTo](a: Double): B = ConvertableTo[B].fromDouble(a)
  override def asString(a: Double): String = a.toString
}

trait ConvertableFromLong extends ConvertableFrom[Long] {
  override def toByte(a: Long): Byte = a.toByte
  override def toInt(a: Long): Int = a.toInt
  override def toLong(a: Long): Long = a
  override def toFloat(a: Long): Float = a.toFloat
  override def toDouble(a: Long): Double = a.toDouble
  override def toType[B: ConvertableTo](a: Long): B = ConvertableTo[B].fromLong(a)
  override def asString(a: Long): String = a.toString
}

trait ConvertableFromInt extends ConvertableFrom[Int] {
  override def toType[B: ConvertableTo](a: Int): B = ConvertableTo[B].fromInt(a)
  override def toByte(a: Int): Byte = a.toByte
  override def toInt(a: Int): Int = a
  override def toLong(a: Int): Long = a.toLong
  override def toFloat(a: Int): Float = a.toFloat
  override def toDouble(a: Int): Double = a.toDouble
  override def asString(a: Int): String = a.toString
}

trait ConvertableFromByte extends ConvertableFrom[Byte] {
  override def toByte(a: Byte): Byte = a
  override def toInt(a: Byte): Int = a.toInt
  override def toLong(a: Byte): Long = a.toLong
  override def toFloat(a: Byte): Float = a.toFloat
  override def toDouble(a: Byte): Double = a.toDouble
  override def toType[B: ConvertableTo](a: Byte): B = ConvertableTo[B].fromByte(a)
  override def asString(a: Byte): String = a.toString
}
