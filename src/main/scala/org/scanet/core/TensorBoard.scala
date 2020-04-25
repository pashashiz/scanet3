package org.scanet.core

import java.nio.file.{Files, Path, Paths}

import com.google.protobuf.ByteString
import org.scanet.native.TfRecords
import org.tensorflow.util.Event
import java.net.InetAddress
import java.time.Instant

object TensorBoard {

  def write[A](ops: List[Output[_]], dir: Path): Unit = {
    val (graph, _) = Session.compileN(ops)
    val version = Event.newBuilder()
      .setFileVersion("brain.Event:2")
      .build()
    val graphDef = Event.newBuilder()
      .setGraphDef(ByteString.copyFrom(graph.toGraphDef))
      .build()
    val computerName = InetAddress.getLocalHost.getHostName
    val file = s"events.out.tfevents.${Instant.now().getEpochSecond}.$computerName"
    TfRecords.of(List(version, graphDef))
      .writeTo(Files.newOutputStream(dir.resolve(file)))
  }
}
