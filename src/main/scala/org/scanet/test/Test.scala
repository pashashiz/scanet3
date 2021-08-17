package org.scanet.test

import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.impl.buffer.nio.NioDataBufferFactory
import org.tensorflow.proto.framework.DataType
import org.tensorflow.types._
import org.tensorflow.{Graph, Session}

import java.nio.FloatBuffer

object Test {

  def main(args: Array[String]): Unit = {
    val graph = new Graph()
    val ba = FloatBuffer.allocate(1)
    ba.put(3.0f)
    ba.rewind()
    val bb = FloatBuffer.allocate(1)
    bb.put(2.0f)
    bb.rewind()
    val a = graph
      .opBuilder("Const", "a")
      .setAttr("dtype", DataType.DT_FLOAT)
      .setAttr("value", TFloat32.tensorOf(Shape.scalar(), NioDataBufferFactory.create(ba)))
      .build
    val b = graph
      .opBuilder("Const", "b")
      .setAttr("dtype", DataType.DT_FLOAT)
      .setAttr("value", TFloat32.tensorOf(Shape.scalar(), NioDataBufferFactory.create(bb)))
      .build
    graph.opBuilder("Add", "z").addInput(a.output(0)).addInput(b.output(0)).build()
    val s = new Session(graph)
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
