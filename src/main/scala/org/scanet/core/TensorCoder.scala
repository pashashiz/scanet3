package org.scanet.core

import java.nio.{ByteBuffer, ByteOrder}

import scala.annotation.tailrec

case class TensorBuffer[A: TensorType](buf: ByteBuffer, size: Int) {
   val coder: TensorCoder[A] = TensorType[A].coder
   def read(pos: Int): A = coder.read(buf, pos, size)
   def write(src: Array[A]): Unit = coder.write(src, buf)
   def writeBytes(src: Array[Byte]): Unit = buf.put(src).rewind()
}

trait TensorCoder[A] {

   def write(src: Array[A], dest: ByteBuffer): Unit
   def read(src: ByteBuffer, pos: Int, bufSize: Int): A
   def sizeOf(src: Array[A]): Int = src.length
}

class FloatTensorCoder extends TensorCoder[Float] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Float = src.getFloat(pos * 4)

   override def write(src: Array[Float], dest: ByteBuffer): Unit = {
      dest.asFloatBuffer().put(src).rewind()
   }
}

class DoubleTensorCoder extends TensorCoder[Double] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Double = src.getDouble(pos * 8)

   override def write(src: Array[Double], dest: ByteBuffer): Unit = {
      dest.asDoubleBuffer().put(src).rewind()
   }
}

class IntTensorCoder extends TensorCoder[Int] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Int = src.getInt(pos * 4)

   override def write(src: Array[Int], dest: ByteBuffer): Unit = {
      dest.asIntBuffer().put(src).rewind()
   }
}

class LongTensorCoder extends TensorCoder[Long] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Long = src.getLong(pos * 8)

   override def write(src: Array[Long], dest: ByteBuffer): Unit = {
      dest.asLongBuffer().put(src).rewind()
   }
}

class ByteTensorCoder extends TensorCoder[Byte] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Byte = src.get(pos)

   override def write(src: Array[Byte], dest: ByteBuffer): Unit = {
      dest.put(src).rewind()
   }
}

class BooleanTensorCoder extends TensorCoder[Boolean] {

   override def read(buf: ByteBuffer, pos: Int, bufSize: Int): Boolean = buf.get(pos) == 1

   override def write(src: Array[Boolean], dest: ByteBuffer): Unit = {
      dest.put(src.map(b => if (b) 1.toByte else 0.toByte)).rewind()
   }
}

class StringTensorCoder extends TensorCoder[String] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): String = {
      @tailrec
      def loop(idx: Int, offset: Int, dst: Array[Byte]): String =
         if (idx == dst.length) new String(dst, "UTF-8")
         else {
            dst(idx) = src.get(offset + idx)
            loop(idx + 1, offset, dst)
         }
      def offset(pos: Int) =
         if (pos == bufSize) src.limit
         else src.getLong(pos * 8).toInt + bufSize * 8

      val curr = offset(pos)
      val next = offset(pos + 1)
      val offsetLen = next - curr
      val len = offsetLen - ubytes(offsetLen - ubytes(offsetLen))
      loop(0, curr + ubytes(len), Array.ofDim[Byte](len))
   }

   override def write(src: Array[String], dest: ByteBuffer): Unit = {
      src.foldLeft(0)((offset, str) => {
         dest.putLong(offset)
         offset + strLen(str)
      })
      src.foreach(str => {
         dest.put(asUBytes(str.length, dest.order))
         dest.put(str.getBytes)
      })
      dest.rewind()
   }

   override def sizeOf(array: Array[String]): Int = array.length * 8 + array.map(strLen).sum

   private def strLen(str: String): Int = ubytes(str.length) + str.length

   private def ubytes(num: Long): Int = {
      @tailrec
      def loop(i: Long, len: Int): Int =
         if (i <= Byte.MaxValue) len
         else loop(i >> 7, len + 1)
      loop(num, 1)
   }

   private def asUBytes(int: Int, order: ByteOrder): Array[Byte] = {
      val len = ubytes(int)
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