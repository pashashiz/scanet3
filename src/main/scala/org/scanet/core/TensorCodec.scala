package org.scanet.core

import org.bytedeco.javacpp.{BytePointer, Pointer}
import org.scanet.native.RawTensors
import org.tensorflow.RawTensor
import org.tensorflow.internal.c_api.TF_TString
import org.tensorflow.internal.c_api.global.tensorflow.{TF_TString_Copy, TF_TString_GetDataPointer, TF_TString_GetSize}

import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets.UTF_8

case class TensorBuffer[A: TensorType](buf: ByteBuffer) {
  val codec: TensorCodec[A] = TensorType[A].codec
  def read(pos: Int): A = codec.read(buf, pos)
  def write(src: Array[A]): Unit = codec.write(src, buf)
  def writeBytes(src: Array[Byte]): Unit = buf.put(src).rewind()
}

object TensorBuffer {
  def of[A: TensorType](rawTensor: RawTensor): TensorBuffer[A] =
    TensorBuffer[A](RawTensors.nativeMemoryOf(rawTensor))
}

trait TensorCodec[A] {

  def write(src: Array[A], dest: ByteBuffer): Unit
  def read(src: ByteBuffer, pos: Int): A
  def sizeOf(elements: Int): Int
}

object FloatTensorCodec extends TensorCodec[Float] {

  override def read(src: ByteBuffer, pos: Int): Float = src.getFloat(pos * 4)

  override def write(src: Array[Float], dest: ByteBuffer): Unit = {
    dest.asFloatBuffer().put(src).rewind()
  }

  def sizeOf(elements: Int): Int = elements * 4
}

object DoubleTensorCodec extends TensorCodec[Double] {

  override def read(src: ByteBuffer, pos: Int): Double = src.getDouble(pos * 8)

  override def write(src: Array[Double], dest: ByteBuffer): Unit = {
    dest.asDoubleBuffer().put(src).rewind()
  }

  def sizeOf(elements: Int): Int = elements * 8
}

object IntTensorCodec extends TensorCodec[Int] {

  override def read(src: ByteBuffer, pos: Int): Int = src.getInt(pos * 4)

  override def write(src: Array[Int], dest: ByteBuffer): Unit = {
    dest.asIntBuffer().put(src).rewind()
  }

  def sizeOf(elements: Int): Int = elements * 4
}

object LongTensorCodec extends TensorCodec[Long] {

  override def read(src: ByteBuffer, pos: Int): Long = src.getLong(pos * 8)

  override def write(src: Array[Long], dest: ByteBuffer): Unit = {
    dest.asLongBuffer().put(src).rewind()
  }

  def sizeOf(elements: Int): Int = elements * 8
}

object ByteTensorCodec extends TensorCodec[Byte] {

  override def read(src: ByteBuffer, pos: Int): Byte = src.get(pos)

  override def write(src: Array[Byte], dest: ByteBuffer): Unit = {
    dest.put(src).rewind()
  }

  def sizeOf(elements: Int): Int = elements
}

object BooleanTensorCodec extends TensorCodec[Boolean] {

  override def read(src: ByteBuffer, pos: Int): Boolean = src.get(pos) == 1

  override def write(src: Array[Boolean], dest: ByteBuffer): Unit = {
    dest.put(src.map(b => if (b) 1.toByte else 0.toByte)).rewind()
  }

  def sizeOf(elements: Int): Int = elements
}

object StringTensorCodec extends TensorCodec[String] {

  private val pointerSize = {
    // Ensure that TensorFlow native library and classes are ready to be used
    // so the TF_TString size would be 24 instead of 8 bytes
    Class.forName("org.tensorflow.TensorFlow")
    Pointer.sizeof(classOf[TF_TString])
  }

  override def read(src: ByteBuffer, pos: Int): String = {
    require(
      src.capacity() >= pointerSize * pos,
      s"string tensor is out of bound, tried to read $pos out of ${src.capacity() / pointerSize}")
    val strings = new TF_TString(new Pointer(src))
    val string = strings.getPointer(pos)
    val pointer = TF_TString_GetDataPointer(string).capacity(TF_TString_GetSize(string))
    pointer.getString(UTF_8)
  }

  override def write(src: Array[String], dest: ByteBuffer): Unit = {
    require(
      dest.capacity() >= src.length,
      s"string tensor is out of bound, cannot write ${src.length} elements into" +
      s"a string tensor of ${dest.capacity() / pointerSize} size")
    val strings = new TF_TString(new Pointer(dest))
    src.indices.foreach { i =>
      val bytes = src(i).getBytes(UTF_8)
      val bp = new BytePointer(bytes: _*)
      TF_TString_Copy(strings.getPointer(i), bp, bytes.length)
      bp.deallocate()
    }
  }

  override def sizeOf(elements: Int): Int = elements * pointerSize
}