package org.scanet.native

import java.nio.ByteBuffer

import org.scanet.core.{Shape, TensorType}
import org.tensorflow.{DataType, Tensor => NativeTensor}

object NativeTensorOps {

  def buffer[A: TensorType](tensor: NativeTensor[A]): ByteBuffer = {
    val method = classOf[NativeTensor[A]].getDeclaredMethod("buffer")
    method.setAccessible(true)
    method.invoke(tensor).asInstanceOf[ByteBuffer]
  }

  def allocate[A: TensorType](shape: Shape): NativeTensor[A] = {
    val method = classOf[NativeTensor[A]].getDeclaredMethod(
      "allocateForBuffer",
      classOf[DataType],
      classOf[Array[Long]],
      Integer.TYPE)
    method.setAccessible(true)
     method.invoke(
       null,
       TensorType[A].tag,
       shape.toLongArray,
       shape.power.asInstanceOf[AnyRef]
     ).asInstanceOf[NativeTensor[A]]
  }
}
