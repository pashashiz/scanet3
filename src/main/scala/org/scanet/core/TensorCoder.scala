package org.scanet.core

import java.nio.{ByteBuffer, ByteOrder}

import scala.annotation.tailrec

case class TensorBuffer[A: TensorType](buf: ByteBuffer, size: Int) {
   val coder: TensorCoder[A] = TensorType[A].coder
   def read(pos: Int): A = coder.read(buf, pos, size)
   def write(array: Array[A]): Unit = coder.write(buf, array)
}

trait TensorCoder[A] {
   def write(buf: ByteBuffer, array: Array[A]): Unit
   def read(buf: ByteBuffer, pos: Int, bufSize: Int): A
   def size(array: Array[A]): Int = array.length
}

class FloatTensorCoder extends TensorCoder[Float] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Float = buf.getFloat(pos * 4)

   override def write(buf: ByteBuffer, array: Array[Float]): Unit = {
      buf.asFloatBuffer().put(array).rewind()
   }
}

class DoubleTensorCoder extends TensorCoder[Double] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Double = buf.getDouble(pos * 8)

   override def write(buf: ByteBuffer, array: Array[Double]): Unit = {
      buf.asDoubleBuffer().put(array).rewind()
   }
}

class IntTensorCoder extends TensorCoder[Int] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Int = buf.getInt(pos * 4)

   override def write(buf: ByteBuffer, array: Array[Int]): Unit = {
      buf.asIntBuffer().put(array).rewind()
   }
}

class LongTensorCoder extends TensorCoder[Long] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Long = buf.getLong(pos * 8)

   override def write(buf: ByteBuffer, array: Array[Long]): Unit = {
      buf.asLongBuffer().put(array).rewind()
   }
}

class ByteTensorCoder extends TensorCoder[Byte] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Byte = buf.get(pos)

   override def write(buf: ByteBuffer, array: Array[Byte]): Unit = {
      buf.put(array).rewind()
   }
}

class BooleanTensorCoder extends TensorCoder[Boolean] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Boolean = buf.get(pos) == 1

   override def write(buf: ByteBuffer, array: Array[Boolean]): Unit = {
      buf.put(array.map(b => if (b) 1.toByte else 0.toByte)).rewind()
   }
}

class StringTensorCoder extends TensorCoder[String] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): String = {
      @tailrec
      def loop(idx: Int, offset: Int, dst: Array[Byte]): String =
         if (idx == dst.length) new String(dst)
         else {
            dst(idx) = buf.get(offset + idx)
            loop(idx + 1, offset, dst)
         }
      def offset(pos: Int) =
         if (pos == bufSize) buf.limit
         else buf.getLong(pos * 8).toInt + bufSize * 8

      val curr = offset(pos)
      val next = offset(pos + 1)
      val offsetLen = next - curr
      val len = offsetLen - bytes(offsetLen - bytes(offsetLen))
      loop(0, curr + bytes(len), Array.ofDim[Byte](len))
   }

   override def write(buf: ByteBuffer, array: Array[String]): Unit = {
      array.foldLeft(0)((offset, str) => {
         buf.putLong(offset)
         offset + strLen(str)
      })
      array.foreach(str => {
         buf.put(asUBytes(str.length, buf.order))
         buf.put(str.getBytes)
      })
      buf.rewind()
   }

   override def size(array: Array[String]): Int = array.length * 8 + array.map(strLen).sum

   private def strLen(str: String): Int = bytes(str.length) + str.length

   private def bytes(num: Long): Int = {
      @tailrec
      def loop(i: Long, len: Int): Int =
         if (i <= Byte.MaxValue) len
         else loop(i >> 7, len + 1)
      loop(num, 1)
   }

   private def asUBytes(int: Int, order: ByteOrder): Array[Byte] = {
      val len = bytes(int)
      val arr = Array.ofDim[Byte](len)
      @tailrec
      def loop(i: Int): Array[Byte] = {
         if (i < 0) arr
         else {
            val idx = if (order == ByteOrder.BIG_ENDIAN) i else len - i - 1
            arr(idx) = (int >>> (i * 8) & 0xFF).toByte
            loop(i - 1)
         }
      }
      loop(len - 1)
   }
}