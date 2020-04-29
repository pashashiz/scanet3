package org.scanet.core

import java.nio.ByteBuffer

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
      val len = next - curr - bytes(next - curr - 1)
      loop(0, curr + bytes(len), Array.ofDim[Byte](len))
   }

   override def write(buf: ByteBuffer, array: Array[String]): Unit = {
      // TODO: this is wrong - it will work only if buffer is empty
      // if we are appending strings - we would need to update offsets header
      array.foldLeft(0)((offset, str) => {
         buf.putLong(offset)
         offset + bytes(str.length) + str.length
      })
      array.foreach(str => {
         if (str.length > Byte.MaxValue) buf.putInt(str.length)
         else buf.put(str.length.toByte)
         buf.put(str.getBytes)
      })
      buf.rewind()
   }

   override def size(array: Array[String]): Int =
      array.length * 8 + array.foldLeft(0)((size, str) => size + bytes(str.length) + str.length)

   def bytes(i: Int): Int = if (i <= Byte.MaxValue) 1 else 4
}