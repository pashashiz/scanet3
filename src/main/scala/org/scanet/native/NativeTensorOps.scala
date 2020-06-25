package org.scanet.native

import java.nio.ByteBuffer

import org.scanet.core.{Shape, TensorType}
import org.tensorflow.{DataType, Shape => NativeShape, Tensor => NativeTensor}

class RichDataType(val dataType: DataType) {

  def code: Int = {
    val c = classOf[DataType].getDeclaredMethod("c")
    c.setAccessible(true)
    c.invoke(dataType).asInstanceOf[Int]
  }
}

object NativeTensorOps {

  def buffer[A: TensorType](tensor: NativeTensor[A]): ByteBuffer = {
    val method = classOf[NativeTensor[A]].getDeclaredMethod("buffer")
    method.setAccessible(true)
    method.invoke(tensor).asInstanceOf[ByteBuffer]
  }

  def allocateElements[A: TensorType](shape: Array[Long], size: Int): NativeTensor[A] = {
    val allocateForBuffer = classOf[NativeTensor[A]].getDeclaredMethod(
      "allocateForBuffer",
      classOf[DataType],
      classOf[Array[Long]],
      Integer.TYPE)
    allocateForBuffer.setAccessible(true)
    allocateForBuffer.invoke(
      null,
      TensorType[A].tag,
      shape,
      size.asInstanceOf[AnyRef]
    ).asInstanceOf[NativeTensor[A]]
  }

  def allocateBytes[A: TensorType](shape: Array[Long], bytes: Long): NativeTensor[A] = {
    val allocate = classOf[NativeTensor[A]].getDeclaredMethod(
      "allocate",
      java.lang.Integer.TYPE, // tag.c()
      classOf[Array[Long]],   // shape
      java.lang.Long.TYPE)    // bytes length
    allocate.setAccessible(true)
    val nativeHandle = allocate.invoke(
      null,
      TensorType[A].code.asInstanceOf[AnyRef],
      shape,
      bytes.asInstanceOf[AnyRef]
    ).asInstanceOf[Long]
    val fromHandle = classOf[NativeTensor[A]].getDeclaredMethod(
      "fromHandle",
      java.lang.Long.TYPE) // nativeHandle
    fromHandle.setAccessible(true)
    fromHandle.invoke(
      null,
      nativeHandle.asInstanceOf[AnyRef]
    ).asInstanceOf[NativeTensor[A]]
  }


  implicit def toNativeShape(shape: Shape): NativeShape = {
    if (shape.isScalar) {
      NativeShape.scalar()
    } else {
      NativeShape.make(shape.dims.head, shape.dims.tail.map(_.toLong): _*)
    }
  }

}