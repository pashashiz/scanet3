package org.scanet.core

import java.nio.ByteBuffer

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.syntax._
import org.tensorflow.{Tensor => NativeTensor}

class NativeTest extends AnyFlatSpec {

  "native string tensor" should "work" in {
    import org.tensorflow.Tensor
    val matrix = Array.ofDim[Byte](4, 0)
    matrix(0) = "this".getBytes("UTF-8")
    matrix(1) = "is fucked up".getBytes("UTF-8")
//    matrix(0)(1) = "is".getBytes("UTF-8")
//    matrix(1)(0) = "a".getBytes("UTF-8")
    val tensor = Tensor.create(matrix, classOf[String])
    val method = classOf[NativeTensor[String]].getDeclaredMethod("buffer")
    method.setAccessible(true)
    val byteBuffer = method.invoke(tensor).asInstanceOf[ByteBuffer]
    val b = Buffer[Byte](byteBuffer)
//    val str = new String(b.toArray, "UTF-8")
//    str.toArray[Char].foreach(println(_))
    println(b.getLong())
    println(b.getLong)
    println(b.getLong)
    println(b.getLong)
    while (b.hasRemaining) {
      println(b.get)
    }
//    val result = Array.ofDim[Byte](2, 2, 0)
//    tensor.copyTo(result)
//    println(result)
  }

  "tensor-board" should "work" in {

  }
}
