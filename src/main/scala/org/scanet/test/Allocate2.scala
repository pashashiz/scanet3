package org.scanet.test

import org.bytedeco.javacpp.{BytePointer, Loader, Pointer}
import org.scanet.core.Shape
import org.tensorflow.internal.c_api.{Deallocator_Pointer_long_Pointer, TF_TString}
import org.tensorflow.internal.c_api.global.tensorflow._

import java.nio.ByteBuffer

object Allocate2 {

  val noopDeallocator = new Deallocator_Pointer_long_Pointer() {
    override def call(data: Pointer, len: Long, arg: Pointer): Unit = {}
  }.retainReference[Deallocator_Pointer_long_Pointer]()

  def main(args: Array[String]): Unit = {
    (0 until 1000).foreach(run)
  }

  def run(i: Int): Unit = {
    val n = 5
    val shape = Shape(n)
    val strings = new TF_TString(n)
    (0L until n).foreach { j =>
      val data = s"$i-$j"
      val bytes = data.getBytes("UTF-8")
      TF_TString_Init(strings.getPointer(j))
      TF_TString_Copy(strings.getPointer(j), new BytePointer(bytes: _*), bytes.length)
    }
    val pointerSize = 8 // Loader.sizeof(classOf[TF_TString]) gives 24 for some reason...
    val tensor = TF_NewTensor(
      TF_STRING,
      shape.toLongArray,
      shape.rank,
      strings,
      shape.power * pointerSize,
      noopDeallocator,
      null)

    val pointer: Pointer = TF_TensorData(tensor).capacity(TF_TensorByteSize(tensor))
    val buffer = pointer.asBuffer().asInstanceOf[ByteBuffer]

    val strings1 = new TF_TString(new Pointer(buffer))
    val tstring1 = strings1.getPointer(1)
    val ptr1 = TF_TString_GetDataPointer(tstring1).capacity(TF_TString_GetSize(tstring1))
    println(new String(ptr1.getStringBytes))

    TF_DeleteTensor(tensor)
    (0L until n).foreach { i =>
      TF_TString_Dealloc(strings1.position(i))
    }
    tensor.setNull()
  }
}
