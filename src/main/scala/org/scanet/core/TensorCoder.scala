package org.scanet.core

import java.nio.ByteBuffer

case class TensorBuffer[A: TensorType](buf: ByteBuffer) {
   val coder: TensorCoder[A] = TensorType[A].coder
   def size: Int = coder.size(buf)
   def read(pos: Int): A = coder.read(buf, pos)
   def write(array: Array[A]): Unit = coder.write(buf, array)
}

trait TensorCoder[A] {
   def write(buf: ByteBuffer, array: Array[A]): Unit
   def read(buf: ByteBuffer, pos: Int): A
   def size(buf: ByteBuffer): Int
}

class FloatTensorCoder extends TensorCoder[Float] {

   override def read(buf: ByteBuffer, pos: Int): Float = buf.getFloat(pos * 4)

   override def size(buf: ByteBuffer): Int = (buf.limit() - buf.position()) / 4

   override def write(buf: ByteBuffer, array: Array[Float]): Unit = {
      buf.asFloatBuffer().put(array).rewind()
   }
}

class DoubleTensorCoder extends TensorCoder[Double] {

   override def read(buf: ByteBuffer, pos: Int): Double = buf.getDouble(pos * 8)

   override def size(buf: ByteBuffer): Int = (buf.limit() - buf.position()) / 8

   override def write(buf: ByteBuffer, array: Array[Double]): Unit = {
      buf.asDoubleBuffer().put(array).rewind()
   }
}

class IntTensorCoder extends TensorCoder[Int] {

   override def read(buf: ByteBuffer, pos: Int): Int = buf.getInt(pos * 4)

   override def size(buf: ByteBuffer): Int = (buf.limit() - buf.position()) / 4

   override def write(buf: ByteBuffer, array: Array[Int]): Unit = {
      buf.asIntBuffer().put(array).rewind()
   }
}

class LongTensorCoder extends TensorCoder[Long] {

   override def read(buf: ByteBuffer, pos: Int): Long = buf.getLong(pos * 8)

   override def size(buf: ByteBuffer): Int = (buf.limit() - buf.position()) / 8

   override def write(buf: ByteBuffer, array: Array[Long]): Unit = {
      buf.asLongBuffer().put(array).rewind()
   }
}

class ByteTensorCoder extends TensorCoder[Byte] {

   override def read(buf: ByteBuffer, pos: Int): Byte = buf.get(pos)

   override def size(buf: ByteBuffer): Int = buf.limit() - buf.position()

   override def write(buf: ByteBuffer, array: Array[Byte]): Unit = {
      buf.put(array).rewind()
   }
}

class BooleanTensorCoder extends TensorCoder[Boolean] {

   override def read(buf: ByteBuffer, pos: Int): Boolean = buf.get(pos) == 1

   override def size(buf: ByteBuffer): Int = buf.limit() - buf.position()

   override def write(buf: ByteBuffer, array: Array[Boolean]): Unit = {
      buf.put(array.map(b => if (b) 1.toByte else 0.toByte)).rewind()
   }
}

// todo: Yurii, please, implement
class StringTensorCoder extends TensorCoder[String] {

   override def read(buf: ByteBuffer, pos: Int): String = ???

   override def size(buf: ByteBuffer): Int = ???

   override def write(buf: ByteBuffer, array: Array[String]): Unit = ???
}