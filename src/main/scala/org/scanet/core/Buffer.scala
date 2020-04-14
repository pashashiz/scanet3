package org.scanet.core

import java.nio.{ByteBuffer, DoubleBuffer, FloatBuffer, IntBuffer, LongBuffer, Buffer => JavaBuffer}

import org.scanet.math
import org.scanet.math.Numeric._
import org.scanet.math.Numeric.syntax._

import scala.{specialized => sp}

class Buffer[@sp A: math.Numeric](val original: JavaBuffer) extends Comparable[Buffer[A]] {

  private def asFloat: FloatBuffer = original.asInstanceOf[FloatBuffer]
  private def asDouble: DoubleBuffer = original.asInstanceOf[DoubleBuffer]
  private def asLong: LongBuffer = original.asInstanceOf[LongBuffer]
  private def asInt: IntBuffer = original.asInstanceOf[IntBuffer]
  private def asByte: ByteBuffer = original.asInstanceOf[ByteBuffer]

  def slice: Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.slice(); this
    case DoubleTag => asDouble.slice(); this
    case LongTag => asLong.slice(); this
    case IntTag => asInt.slice(); this
    case ByteTag => asByte.slice(); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def duplicate: Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.duplicate(); this
    case DoubleTag => asDouble.duplicate(); this
    case LongTag => asLong.duplicate(); this
    case IntTag => asInt.duplicate(); this
    case ByteTag => asByte.duplicate(); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def asReadOnlyBuffer: Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.asReadOnlyBuffer(); this
    case DoubleTag => asDouble.asReadOnlyBuffer(); this
    case LongTag => asLong.asReadOnlyBuffer(); this
    case IntTag => asInt.asReadOnlyBuffer(); this
    case ByteTag => asByte.asReadOnlyBuffer(); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def get: A = math.Numeric[A].tag match {
    case FloatTag => asFloat.get().asInstanceOf[A]
    case DoubleTag => asDouble.get().asInstanceOf[A]
    case LongTag => asLong.get().asInstanceOf[A]
    case IntTag => asInt.get().asInstanceOf[A]
    case ByteTag => asByte.get().asInstanceOf[A]
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def put(f: A): Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.put(f.asInstanceOf[Float]); this
    case DoubleTag => asDouble.put(f.asInstanceOf[Double]); this
    case LongTag => asLong.put(f.asInstanceOf[Long]); this
    case IntTag => asInt.put(f.asInstanceOf[Int]); this
    case ByteTag => asByte.put(f.asInstanceOf[Byte]); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def get(index: Int): A = math.Numeric[A].tag match {
    case FloatTag => asFloat.get(index).asInstanceOf[A]
    case DoubleTag => asDouble.get(index).asInstanceOf[A]
    case LongTag => asLong.get(index).asInstanceOf[A]
    case IntTag => asInt.get(index).asInstanceOf[A]
    case ByteTag => asByte.get(index).asInstanceOf[A]
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def put(index: Int, f: A): Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.put(index, f.asInstanceOf[Float]); this
    case DoubleTag => asDouble.put(index, f.asInstanceOf[Double]); this
    case LongTag => asLong.put(index, f.asInstanceOf[Long]); this
    case IntTag => asInt.put(index, f.asInstanceOf[Int]); this
    case ByteTag => asByte.put(index, f.asInstanceOf[Byte]); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def get(dst: Array[A]): Buffer[A] = get(dst, 0, dst.length)

  def put(src: Buffer[A]): Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.put(src.asFloat); this
    case DoubleTag => asDouble.put(src.asDouble); this
    case LongTag => asLong.put(src.asLong); this
    case IntTag => asInt.put(src.asInt); this
    case ByteTag => asByte.put(src.asByte); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def put(src: Array[A], offset: Int, length: Int): Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.put(src.asInstanceOf[Array[Float]], offset, length); this
    case DoubleTag => asDouble.put(src.asInstanceOf[Array[Double]], offset, length); this
    case LongTag => asLong.put(src.asInstanceOf[Array[Long]], offset, length); this
    case IntTag => asInt.put(src.asInstanceOf[Array[Int]], offset, length); this
    case ByteTag => asByte.put(src.asInstanceOf[Array[Byte]], offset, length); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def putAll(src: Array[A]): Buffer[A] = put(src, 0, src.length)

  def compact: Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.compact(); this
    case DoubleTag => asDouble.compact(); this
    case LongTag => asLong.compact(); this
    case IntTag => asInt.compact(); this
    case ByteTag => asByte.compact(); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def get(dst: Array[A], offset: Int, length: Int): Buffer[A] = math.Numeric[A].tag match {
    case FloatTag => asFloat.get(dst.asInstanceOf[Array[Float]], offset, length); this
    case DoubleTag => asDouble.get(dst.asInstanceOf[Array[Double]], offset, length); this
    case LongTag => asLong.get(dst.asInstanceOf[Array[Long]], offset, length); this
    case IntTag => asInt.get(dst.asInstanceOf[Array[Int]], offset, length); this
    case ByteTag => asByte.get(dst.asInstanceOf[Array[Byte]], offset, length); this
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }

  def capacity: Int = original.capacity()

  def position: Int = original.position()

  def position(newPosition: Int): Buffer[A] = {original.position(newPosition); this}

  def limit: Int = original.limit()

  def limit(newLimit: Int): Buffer[A] = {original.limit(newLimit); this}

  def mark: Buffer[A] = {original.mark(); this}

  def reset: Buffer[A] = {original.reset(); this}

  def clear: Buffer[A] = {original.clear(); this}

  def flip: Buffer[A] = {original.flip(); this}

  def rewind: Buffer[A] = {original.rewind(); this}

  def remaining: Int = original.remaining()

  def hasRemaining: Boolean = original.hasRemaining

  def isReadOnly: Boolean = original.isReadOnly

  def hasArray: Boolean = original.hasArray

  def toArray: Array[A] = {
    val originalPosition = position
    val array = Array.ofDim[A](limit - position)(math.Numeric[A].classTag)
    while (hasRemaining) {
      array(position - originalPosition) = get
    }
    rewind
    array
  }

  def toStream: Stream[A] = {
    def next(index: Int): Stream[A] = {
      if (limit == index) Stream.empty
      else get(index) #:: next(index + 1)
    }
    next(position)
  }

  def arrayOffset: Int = original.arrayOffset()

  def isDirect: Boolean = original.isDirect

  override def toString: String = s"Buffer[${math.Numeric[A].show}](capacity=$capacity, position=$position, limit=$limit, direct=$isDirect)" + show()

  def show(n: Int = 20): String = {
    val elements = toStream.take(n).mkString(", ")
    "[" + elements + (if (n < limit) "..." else "") + "]"
  }

  override def hashCode: Int = original.hashCode()

  override def equals(other: Any): Boolean = other match {
    case other: Buffer[A] => original == other.original
    case _ => false
  }

  override def compareTo(that: Buffer[A]): Int = math.Numeric[A].tag match {
    case FloatTag => asFloat.compareTo(that.asFloat)
    case DoubleTag => asDouble.compareTo(that.asDouble)
    case LongTag => asLong.compareTo(that.asLong)
    case IntTag => asInt.compareTo(that.asInt)
    case ByteTag => asByte.compareTo(that.asByte)
    case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
  }
}

object Buffer {

  def apply[@sp A: math.Numeric](original: JavaBuffer): Buffer[A] = new Buffer[A](original)

  def from[@sp A: math.Numeric](buffer: ByteBuffer): Buffer[A] = {
    math.Numeric[A].tag match {
      case FloatTag => Buffer(buffer.asFloatBuffer())
      case DoubleTag => Buffer(buffer.asDoubleBuffer())
      case LongTag => Buffer(buffer.asLongBuffer())
      case IntTag => Buffer(buffer.asIntBuffer())
      case ByteTag => Buffer(buffer)
      case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
    }
  }

  def allocate[@sp A: math.Numeric](capacity: Int): Buffer[A] = {
    math.Numeric[A].tag match {
      case FloatTag => Buffer(FloatBuffer.allocate(capacity))
      case DoubleTag => Buffer(DoubleBuffer.allocate(capacity))
      case LongTag => Buffer(LongBuffer.allocate(capacity))
      case IntTag => Buffer(IntBuffer.allocate(capacity))
      case ByteTag => Buffer(ByteBuffer.allocate(capacity))
      case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
    }
  }

  def wrap[@sp A: math.Numeric](array: Array[A], offset: Int, length: Int): Buffer[A] = {
    math.Numeric[A].tag match {
      case FloatTag => Buffer[A](FloatBuffer.wrap(array.asInstanceOf[Array[Float]], offset, length))
      case DoubleTag => Buffer[A](DoubleBuffer.wrap(array.asInstanceOf[Array[Double]], offset, length))
      case LongTag => Buffer[A](LongBuffer.wrap(array.asInstanceOf[Array[Long]], offset, length))
      case IntTag => Buffer[A](IntBuffer.wrap(array.asInstanceOf[Array[Int]], offset, length))
      case ByteTag => Buffer[A](ByteBuffer.wrap(array.asInstanceOf[Array[Byte]], offset, length))
      case _ => error(s"Buffer[${math.Numeric[A].show}] is not supported")
    }
  }

  def wrap[@sp A: math.Numeric](array: Array[A]): Buffer[A] = wrap(array, 0, array.length)

  def tabulate[@sp A: math.Numeric](capacity: Int)(f: Int => A): Buffer[A] = {
    (0 until capacity).foldLeft(Buffer.allocate[A](capacity))((b, i) => b.put(i, f(i)))
  }

  // ## Implicit conversions ##
  // TBuffer -> Buffer[T]
  // Buffer[T] -> TBuffer

  implicit def fromFloatBufferToBuffer(buffer: FloatBuffer): Buffer[Float] = new Buffer(buffer)
  implicit def fromBufferToFloatBuffer(buffer: Buffer[Float]): FloatBuffer = buffer.asFloat

  implicit def fromDoubleBufferToBuffer(buffer: DoubleBuffer): Buffer[Double] = new Buffer(buffer)
  implicit def fromBufferToDoubleBuffer(buffer: Buffer[Double]): DoubleBuffer = buffer.asDouble

  implicit def fromLongBufferToBuffer(buffer: LongBuffer): Buffer[Long] = new Buffer(buffer)
  implicit def fromBufferToLongBuffer(buffer: Buffer[Long]): LongBuffer = buffer.asLong

  implicit def fromIntBufferToBuffer(buffer: IntBuffer): Buffer[Int] = new Buffer(buffer)
  implicit def fromBufferToIntBuffer(buffer: Buffer[Int]): IntBuffer = buffer.asInt

  implicit def fromByteBufferToBuffer(buffer: ByteBuffer): Buffer[Byte] = new Buffer(buffer)
  implicit def fromBufferToByteBuffer(buffer: Buffer[Byte]): ByteBuffer = buffer.asByte

}
