package org.scanet.native

import org.bytedeco.javacpp.{Pointer, PointerScope}
import org.scanet.core.TensorType.StringType
import org.scanet.core.TensorType.syntax._
import org.scanet.core.{Shape, TensorType, Using}
import org.tensorflow.RawTensor
import org.tensorflow.internal.c_api.global.tensorflow._
import org.tensorflow.internal.c_api.{Deallocator_Pointer_long_Pointer, TF_TString, TF_Tensor}
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
    val pointer: Pointer = TF_TensorData(native).capacity(rawTensor.numBytes())
    pointer.asBuffer().asInstanceOf[ByteBuffer]
  }

  def allocate[A: TensorType](shape: Shape): RawTensor = WithinPointer.allocate[A](shape)

  implicit def toNativeShape(shape: Shape): NativeShape =
    NativeShape.of(shape.toLongArray: _*)
}

object WithinPointer extends Pointer {

  private val rawTensorClass = classOf[RawTensor]
  private val rawTensorCtor = {
    val ctor = rawTensorClass.getDeclaredConstructors.head
    ctor.setAccessible(true)
    ctor
  }
  private val rawTensorHandle = {
    val tensorHandle = rawTensorClass.getDeclaredField("tensorHandle")
    tensorHandle.setAccessible(true)
    tensorHandle
  }
  private val rawTensorScope = {
    val tensorScope = rawTensorClass.getDeclaredField("tensorScope")
    tensorScope.setAccessible(true)
    tensorScope
  }
  private val pointerClass = classOf[Pointer]
  private val pointerDeallocatorMethod = {
    val method = pointerClass.getDeclaredMethod("deallocator", classOf[Pointer.Deallocator])
    method.setAccessible(true)
    method
  }

  def allocate[A: TensorType](shape: Shape): RawTensor = {
    val nativeTensor = TensorType[A].tag match {
      case StringType => allocateString(shape)
      case _          => allocatePrimitive[A](shape)
    }
    // we use scope here, so if someone would call close() on a RawTensor we will
    // also clean TF_Tensor eagerly
    Using.resource(new PointerScope) { scope =>
      scope.attach(nativeTensor)
      val nativeShape = RawTensors.toNativeShape(shape)
      val typeInfo = TensorTypeRegistry.find(TensorType[A].jtag)
      val rawTensor = rawTensorCtor.newInstance(typeInfo, nativeShape).asInstanceOf[RawTensor]
      rawTensorHandle.set(rawTensor, nativeTensor)
      rawTensorScope.set(rawTensor, scope.extend)
      rawTensor
    }
  }

  def allocatePrimitive[A: TensorType](shape: Shape): TF_Tensor = {
    val ttype = TensorType[A]
    val nativeTensor =
      TF_AllocateTensor(ttype.code, shape.toLongArray, shape.rank, ttype.coder.sizeOf(shape.power))
    require(nativeTensor != null, s"cannot allocate a tensor with shape $shape")
    pointerDeallocatorMethod.invoke(nativeTensor, new PrimitiveTensorDeallocator(nativeTensor))
    nativeTensor
  }

  def allocateString(shape: Shape): TF_Tensor = {
    val ttype = TensorType[String]
    val size = shape.power
    // there is a default deallocator assign to strings array, we should keep a reference so it is not GC-ed
    val strings = new TF_TString(size)
    // we MUST fill the allocated string with an empty value, otherwise the JVM will crash later
    (0L until size).foreach(i => TF_TString_Init(strings.position(i)))
    val nativeTensor = TF_NewTensor(
      TF_STRING,
      shape.toLongArray,
      shape.rank,
      strings,
      ttype.coder.sizeOf(size),
      noopDeallocator,
      null)
    require(nativeTensor != null, s"cannot allocate a tensor with shape $shape")
    pointerDeallocatorMethod.invoke(
      nativeTensor,
      new StringTensorDeallocator(nativeTensor, strings))
    nativeTensor
  }

  // IMPORTANT - these deallocators are not collected by default Pointer GC thread, instead
  // they will be collected by our own tensor GC (see Disposable)
  // need to think how to reuse the out of the box Pointer GC (right now the TF_Tensor is nor reachable yet)

  private class PrimitiveTensorDeallocator(tensor: TF_Tensor)
      extends TF_Tensor(tensor)
      with Pointer.Deallocator {
    override def deallocate(): Unit = {
      if (!isNull) {
        TF_DeleteTensor(this)
      }
      setNull()
    }
  }

  private class StringTensorDeallocator(tensor: TF_Tensor, strings: TF_TString)
      extends TF_Tensor(tensor)
      with Pointer.Deallocator {
    override def deallocate(): Unit = {
      if (!isNull) {
//        val size = TF_TensorElementCount(tensor)
        TF_DeleteTensor(this)
//        (0L until size).foreach(i => TF_TString_Dealloc(strings.position(i)))
//        strings.deallocate()
      }
      setNull()
    }
  }

  private val noopDeallocator: Deallocator_Pointer_long_Pointer =
    new Deallocator_Pointer_long_Pointer() {
      override def call(data: Pointer, len: Long, arg: Pointer): Unit = {}
    }.retainReference[Deallocator_Pointer_long_Pointer]()
}
