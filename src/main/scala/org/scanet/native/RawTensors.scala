package org.scanet.native

import org.bytedeco.javacpp.Pointer
import org.scanet.core.{Shape, TensorType}
import org.tensorflow.RawTensor
import org.tensorflow.internal.c_api.TF_Tensor
import org.tensorflow.internal.c_api.global.tensorflow.{TF_TensorByteSize, TF_TensorData}
import org.tensorflow.ndarray.{Shape => NativeShape}

import java.nio.ByteBuffer

object RawTensors {

  /** Get access to a native memory allocated by a [[RawTensor]].
    * The memory is mutable and if changed the tensor will use the updated version.
    * In case the tensor is deallocated the buffer might point into the memory
    * used by other objects, there the danger comes, need to be extremely careful
    * @param rawTensor raw tensor (which is just a wrapper to C tensor via JNI)
    * @return
    */
  def nativeMemoryOf(rawTensor: RawTensor): ByteBuffer = {
    // todo: fix seg fault
    val tensorHandle = classOf[RawTensor].getDeclaredField("tensorHandle")
    tensorHandle.setAccessible(true)
    val native = tensorHandle.get(rawTensor).asInstanceOf[TF_Tensor]
    val pointer: Pointer = TF_TensorData(native).capacity(TF_TensorByteSize(native))
    pointer.asBuffer().asInstanceOf[ByteBuffer]
  }

  def allocate[A: TensorType](shape: Array[Long], bytes: Long): RawTensor = {
    val allocate = classOf[RawTensor].getDeclaredMethod(
      "allocate",
      classOf[Class[_]],
      classOf[NativeShape],
      java.lang.Long.TYPE)
    allocate.setAccessible(true)
    allocate
      .invoke(null, TensorType[A].jtag, NativeShape.of(shape: _*), bytes.asInstanceOf[AnyRef])
      .asInstanceOf[RawTensor]
  }

  implicit def toNativeShape(shape: Shape): NativeShape =
    if (shape.isScalar)
      NativeShape.scalar()
    else
      NativeShape.of(shape.dims.map(_.toLong): _*)
}
