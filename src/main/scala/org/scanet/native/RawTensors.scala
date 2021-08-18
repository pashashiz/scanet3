package org.scanet.native

import org.bytedeco.javacpp.{Pointer, PointerScope}
import org.scanet.core.{Shape, TensorType, Using}
import org.scanet.native.RawTensors.toNativeShape
import org.tensorflow.RawTensor
import org.tensorflow.internal.c_api.{TF_TString, TF_Tensor}
import org.tensorflow.internal.c_api.global.tensorflow.{
  TF_AllocateTensor,
  TF_DeleteTensor,
  TF_STRING,
  TF_TString_Dealloc,
  TF_TensorByteSize,
  TF_TensorData,
  TF_TensorElementCount,
  TF_TensorType
}
import org.tensorflow.internal.types.registry.TensorTypeRegistry
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
    val tensorHandle = classOf[RawTensor].getDeclaredField("tensorHandle")
    tensorHandle.setAccessible(true)
    val native = tensorHandle.get(rawTensor).asInstanceOf[TF_Tensor]
//    val capacity = TF_TensorByteSize(native)
//    println(s"Accessing native memory of a tensor with capacity $capacity")
    val pointer: Pointer = TF_TensorData(native).capacity(rawTensor.numBytes())
    pointer.asBuffer().asInstanceOf[ByteBuffer]
  }

  @deprecated
  def allocate[A: TensorType](shape: Array[Long], bytes: Long): RawTensor = {
    println(s"Allocating tensor with shape ${Shape.of(shape)} and size $bytes")
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

  def allocate[A: TensorType](shape: Shape, bytes: Long): RawTensor =
    WithinPointer.allocate(shape, bytes)

  implicit def toNativeShape(shape: Shape): NativeShape =
    if (shape.isScalar)
      NativeShape.scalar()
    else
      NativeShape.of(shape.dims.map(_.toLong): _*)
}

object WithinPointer extends Pointer {

  private val rtClass = classOf[RawTensor]
  private val ntClass = classOf[Pointer]
  private val deallocatorMethod = {
    val method = ntClass.getDeclaredMethod("deallocator", classOf[Pointer.Deallocator])
    method.setAccessible(true)
    method
  }

  def allocate[A: TensorType](shape: Shape, bytes: Long): RawTensor = {
    val nativeTensor = TF_AllocateTensor(TensorType[A].code, shape.toLongArray, shape.rank, bytes)
    require(nativeTensor != null, "cannot allocate a tensor")
    deallocatorMethod.invoke(nativeTensor, new TensorDealocator(nativeTensor))
    Using.resource(new PointerScope) { scope =>
      scope.attach(nativeTensor)
      val nativeShape = RawTensors.toNativeShape(shape)
      val typeInfo = TensorTypeRegistry.find(TensorType[A].jtag)
      val ctor = rtClass.getDeclaredConstructors.head
      ctor.setAccessible(true)
      val rawTensor = ctor.newInstance(typeInfo, nativeShape).asInstanceOf[RawTensor]
      val tensorHandle = rtClass.getDeclaredField("tensorHandle")
      tensorHandle.setAccessible(true)
      tensorHandle.set(rawTensor, nativeTensor)
      val tensorScope = rtClass.getDeclaredField("tensorScope")
      tensorScope.setAccessible(true)
      tensorScope.set(rawTensor, scope.extend)
      rawTensor
    }
  }

  class TensorDealocator(val s: TF_Tensor) extends TF_Tensor(s) with Pointer.Deallocator {
    override def deallocate(): Unit = {
      if (!isNull) TF_DeleteTensor(this)
      setNull()
    }
  }
}
