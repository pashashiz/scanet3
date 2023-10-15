package scanet.optimizers

import com.esotericsoftware.kryo.Kryo
import com.twitter.chill._
import org.apache.spark.serializer.KryoRegistrator
import scanet.core.{Shape, Tensor, TensorType}
import scanet.models.TrainedModel_
import org.tensorflow.proto.framework.DataType.DT_STRING
import scanet.core.syntax._

class KryoSerializers extends KryoRegistrator {

  override def registerClasses(kryo: Kryo): Unit = {
    kryo.forClass[Tensor[_]](new TensorSerializer())
    kryo.register(classOf[TrainedModel_[_]])
  }
}

object KryoSerializers {
  val Kryo: Kryo = {
    val kryo = new Kryo()
    new KryoSerializers().registerClasses(kryo)
    kryo
  }
}

class TensorSerializer extends KSerializer[Tensor[_]] {

  override def isImmutable: Boolean = true

  override def write(kryo: Kryo, output: Output, tensor: Tensor[_]): Unit = {
    output.writeInt(tensor.`type`.code)
    output.writeInt(tensor.shape.rank)
    output.writeLongs(tensor.shape.toLongArray)
    tensor.`type`.tag match {
      case DT_STRING =>
        // string tensor has each string as a separate object on heap
        // so we need to write those rather than the pointers stored in tensors
        val typed = tensor.asInstanceOf[Tensor[String]]
        typed.toArray.foreach(string => output.writeString(string))
      case _ =>
        // for primitives we can simply write the data as is
        val bytes = tensor.toBytes
        output.writeInt(bytes.length)
        output.writeBytes(tensor.toBytes)
    }
  }

  override def read(kryo: Kryo, input: Input, clazz: Class[Tensor[_]]): Tensor[_] = {
    val tensorType = TensorType.of(input.readInt())
    val rank = input.readInt()
    val shape = Shape.of(input.readLongs(rank))
    tensorType.tag match {
      case DT_STRING =>
        val strings = Array.tabulate(shape.power)(_ => input.readString())
        Tensor[String](strings, shape)
      case _ =>
        val bytesSize = input.readInt()
        val bytes = input.readBytes(bytesSize)
        Tensor.fromBytesUntyped(tensorType.tag, bytes, shape)
    }
  }
}
