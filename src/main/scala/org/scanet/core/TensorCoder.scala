package org.scanet.core

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8

case class TensorBuffer[A: TensorType](buf: ByteBuffer, size: Int) {
   val coder: TensorCoder[A] = TensorType[A].coder
   def read(pos: Int): A = coder.read(buf, pos, size)
   def write(src: Array[A]): Unit = coder.write(src, buf)
   def writeBytes(src: Array[Byte]): Unit = buf.put(src).rewind()
}

trait TensorCoder[A] {

   def write(src: Array[A], dest: ByteBuffer): Unit
   def read(src: ByteBuffer, pos: Int, bufSize: Int): A
   def sizeOf(src: Array[A]): Int
}

class FloatTensorCoder extends TensorCoder[Float] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Float = src.getFloat(pos * 4)

   override def write(src: Array[Float], dest: ByteBuffer): Unit = {
      dest.asFloatBuffer().put(src).rewind()
   }

   def sizeOf(src: Array[Float]): Int = src.length * 4
}

class DoubleTensorCoder extends TensorCoder[Double] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Double = src.getDouble(pos * 8)

   override def write(src: Array[Double], dest: ByteBuffer): Unit = {
      dest.asDoubleBuffer().put(src).rewind()
   }

   def sizeOf(src: Array[Double]): Int = src.length * 8
}

class IntTensorCoder extends TensorCoder[Int] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Int = src.getInt(pos * 4)

   override def write(src: Array[Int], dest: ByteBuffer): Unit = {
      dest.asIntBuffer().put(src).rewind()
   }

   def sizeOf(src: Array[Int]): Int = src.length * 4
}

class LongTensorCoder extends TensorCoder[Long] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Long = src.getLong(pos * 8)

   override def write(src: Array[Long], dest: ByteBuffer): Unit = {
      dest.asLongBuffer().put(src).rewind()
   }

   def sizeOf(src: Array[Long]): Int = src.length * 8
}

class ByteTensorCoder extends TensorCoder[Byte] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Byte = src.get(pos)

   override def write(src: Array[Byte], dest: ByteBuffer): Unit = {
      dest.put(src).rewind()
   }

   def sizeOf(src: Array[Byte]): Int = src.length
}

class BooleanTensorCoder extends TensorCoder[Boolean] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): Boolean = src.get(pos) == 1

   override def write(src: Array[Boolean], dest: ByteBuffer): Unit = {
      dest.put(src.map(b => if (b) 1.toByte else 0.toByte)).rewind()
   }

   def sizeOf(src: Array[Boolean]): Int = src.length
}

/**
 * String tensors are considered to be char tensor with protocol.
 * [0, 3] 4 bytes: N, num of strings in the tensor in little endian.
 * [(i+1)*4, (i+1)*4+3] 4 bytes: offset of i-th string in little endian, for i from 0 to N-1.
 * [(N+1)*4, (N+1)*4+3] 4 bytes: length of the whole char buffer.
 * [offset(i), offset(i+1) - 1] : content of i-th string.
 * Example of a string tensor:
 * [
 *   2, 0, 0, 0,     # 2 strings.
 *   16, 0, 0, 0,    # 0-th string starts from index 16.
 *   18, 0, 0, 0,    # 1-st string starts from index 18.
 *   18, 0, 0, 0,    # total length of array.
 *   'A', 'B',       # 0-th string [16..17]: "AB"
 * ]                 # 1-th string, empty
 */
class StringTensorCoder extends TensorCoder[String] {

   override def read(src: ByteBuffer, pos: Int, bufSize: Int): String = {
      def offset(pos: Int) = src.getInt((pos + 1) * 4)

      val curr = offset(pos)
      val dst = Array.ofDim[Byte](offset(pos + 1) - curr)
      src.position(curr)
      src.get(dst)
      new String(dst, UTF_8)
   }

   override def write(src: Array[String], dest: ByteBuffer): Unit = {
      // header has to be in little endian?
      val headerLength = (2 + src.length) * 4
      dest.putInt(src.length)
      val bodyLength = src.foldLeft(headerLength)((offset, str) => {
         dest.putInt(offset)
         offset + str.length
      })
      dest.putInt(bodyLength)
      src.foreach(str => dest.put(str.getBytes))
      dest.rewind()
   }

   override def sizeOf(array: Array[String]): Int = (array.length + 2) * 4 + array.map(_.length).sum
}