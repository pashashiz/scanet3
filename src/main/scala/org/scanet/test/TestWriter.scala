package org.scanet.test

import java.nio.FloatBuffer
import java.nio.file.{Files, Paths}

import com.google.protobuf.ByteString
import org.scanet.native.{TfRecord, TfRecords}
import org.tensorflow.util.Event
import org.tensorflow.{DataType, Graph, Session, Tensor}

import scala.reflect.io.File

object TestWriter {

  def main(args: Array[String]): Unit = {
    val graph = new Graph()
    val bBuffer = FloatBuffer.allocate(1)
    bBuffer.put(2.0f)
    bBuffer.rewind()
    val b = graph.opBuilder("Const", "b")
      .setAttr("dtype", DataType.FLOAT)
      .setAttr("value", Tensor.create(Array[Long](1), bBuffer))
      .build
    val tags = graph.opBuilder("Const", "bTag")
      .setAttr("dtype", DataType.STRING)
      .setAttr("value", Tensor.create(Array("b-value".getBytes("UTF-8")), classOf[String]))
      .build
    val bSummary = graph.opBuilder("ScalarSummary", "bSummary")
      .addInput(tags.output(0))
      .addInput(b.output(0))
      .build()
    try {
      val session = new Session(graph)
      try {
        val scalarSummary: Tensor[_] = session.runner.fetch(bSummary.output(0)).run.get(0)
//        println(new String(scalarSummary.bytesValue(), "UTF-8"))
//        println(new String(graph.toGraphDef, "UTF-8"))
        val event1: Event = Event.newBuilder()
          .setFileVersion("brain.Event:2")
          .build()
        val event2: Event = Event.newBuilder()
          .setGraphDef(ByteString.copyFrom(graph.toGraphDef))
          .build()
        val out = Files.newOutputStream(Paths.get("events.out.tfevents.1583612234.Pavlos-MacBook-Pro-2.local"))
        TfRecords.of(List(event1, event2)).writeTo(out)
//        println(new String(event.toByteArray, "UTF-8"))
      } finally if (session != null) session.close()
    }
  }
}
