package org.scanet.optimizers

import com.esotericsoftware.kryo.Kryo
import com.twitter.chill._
import org.apache.spark.serializer.KryoRegistrator
import org.scanet.core.{Shape, Tensor, TensorType}

class TensorSerializer extends KryoRegistrator {

  override def registerClasses(kryo: Kryo): Unit = {
    kryo.forClass[Tensor[_]](
      new KSerializer[Tensor[_]] {

        override def isImmutable: Boolean = true

        override def write(kryo: Kryo, output: Output, tensor: Tensor[_]): Unit = {
          println("WRITE")
          val bytes = tensor.toBytes
          output.writeInt(tensor.`type`.code)
          output.writeInt(tensor.shape.rank)
          output.writeLongs(tensor.shape.toLongArray)
          output.writeInt(bytes.length)
          output.writeBytes(tensor.toBytes)
        }

        override def read(kryo: Kryo, input: Input, clazz: Class[Tensor[_]]): Tensor[_] = {
          println("READ")
          val tensorType = TensorType.of(input.readInt())
          val rank = input.readInt()
          val shape = Shape.of(input.readLongs(rank))
          val bytesSize = input.readInt()
          val bytes = input.readBytes(bytesSize)
          Tensor.fromBytesUntyped(tensorType.tag, bytes, shape)
        }
      })
  }
}
