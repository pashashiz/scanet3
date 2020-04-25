package org.scanet.test

import java.nio.FloatBuffer

import org.tensorflow.{DataType, Graph, Session, Tensor}

object Test {

  def main(args: Array[String]): Unit = {
    val graph = new Graph()
    val ba = FloatBuffer.allocate(1)
    ba.put(3.0f)
    ba.rewind()
    val bb = FloatBuffer.allocate(1)
    bb.put(2.0f)
    bb.rewind()
    val a = graph.opBuilder("Const", "a")
      .setAttr("dtype", DataType.FLOAT)
      .setAttr("value", Tensor.create(Array[Long](), ba))
      .build
    val b = graph.opBuilder("Const", "b")
      .setAttr("dtype", DataType.FLOAT)
      .setAttr("value", Tensor.create(Array[Long](), bb))
      .build
    graph.opBuilder("Add", "z")
      .addInput(a.output(0))
      .addInput(b.output(0))
      .build()
    val s = new Session(graph)
    try {
      val res = s.runner.fetch("z").run.get(0)
      try {
        System.out.println(res.floatValue) // Will print 6.0f
      } finally {
        if (res != null) res.close()
      }
    } finally {
      if (s != null) s.close()
    }
  }
}
