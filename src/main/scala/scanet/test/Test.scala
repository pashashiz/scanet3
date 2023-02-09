package scanet.test

import org.tensorflow.internal.c_api.{AbstractTF_Status, TF_Session, TF_Status}
import org.tensorflow.internal.c_api.global.tensorflow.{TF_DeviceListCount, TF_DeviceListIncarnation, TF_DeviceListName, TF_DeviceListType, TF_SessionListDevices}
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.impl.buffer.nio.NioDataBufferFactory
import org.tensorflow.op.OpScope
import org.tensorflow.proto.framework.DataType
import org.tensorflow.types._
import org.tensorflow.{Graph, Session}

import java.nio.FloatBuffer

object Test {

  def main(args: Array[String]): Unit = {
    val graph = new Graph()
    val scope = new OpScope(graph)
    val ba = FloatBuffer.allocate(1)
    ba.put(3.0f)
    ba.rewind()
    val bb = FloatBuffer.allocate(1)
    bb.put(2.0f)
    bb.rewind()
    val a = graph
      .opBuilder("Const", "a", scope)
      .setAttr("dtype", DataType.DT_FLOAT)
      .setAttr("value", TFloat32.tensorOf(Shape.scalar(), NioDataBufferFactory.create(ba)))
      .build
    val b = graph
      .opBuilder("Const", "b", scope)
      .setAttr("dtype", DataType.DT_FLOAT)
      .setAttr("value", TFloat32.tensorOf(Shape.scalar(), NioDataBufferFactory.create(bb)))
      .build
    graph.opBuilder("Add", "z", scope).addInput(a.output(0)).addInput(b.output(0)).build()
    val s = new Session(graph)

    val nativeHandleField = classOf[Session].getDeclaredField("nativeHandle")
    nativeHandleField.setAccessible(true)
    val tfSession = nativeHandleField.get(s).asInstanceOf[TF_Session]

    val s1 = AbstractTF_Status.newStatus()
    val devices = TF_SessionListDevices(tfSession, s1)
    s1.throwExceptionIfNotOK()

    println(s"Device count: ${TF_DeviceListCount(devices)}")

    val s2 = AbstractTF_Status.newStatus()
    val name = TF_DeviceListName(devices, 0, s2)
    s2.throwExceptionIfNotOK()
    println(name.getString)

    val s3 = AbstractTF_Status.newStatus()
    val tpe = TF_DeviceListType(devices, 0, s2)
    s3.throwExceptionIfNotOK()
    println(tpe.getString)


    try {
      val res = s.runner.fetch("z").run.get(0)
      try {
        System.out.println(res.asRawTensor().data().asFloats().getFloat(0)) // Will print 5.0f
      } finally {
        if (res != null) res.close()
      }
    } finally {
      if (s != null) s.close()
    }
  }
}
