package org.scanet.native

import java.nio.ByteBuffer

import org.scanet.core.{Buffer, Numeric}
import org.scanet.linalg.Shape
import org.tensorflow.{DataType, Tensor => NativeTensor}

import scala.{specialized => sp}

object NativeTensorOps {

  def buffer[@sp A: Numeric](tensor: NativeTensor[A]): Buffer[A] = {
    val method = classOf[NativeTensor[A]].getDeclaredMethod("buffer")
    method.setAccessible(true)
    val byteBuffer = method.invoke(tensor).asInstanceOf[ByteBuffer]
    Buffer.from[A](byteBuffer)
  }

  def allocate[A: Numeric](shape: Shape): NativeTensor[A] = {
    val method = classOf[NativeTensor[A]].getDeclaredMethod(
      "allocateForBuffer",
      classOf[DataType],
      classOf[Array[Long]],
      Integer.TYPE)
    method.setAccessible(true)
     method.invoke(
       null,
       Numeric[A].tag,
       shape.toLongArray,
       shape.power.asInstanceOf[AnyRef]
     ).asInstanceOf[NativeTensor[A]]
  }
}
