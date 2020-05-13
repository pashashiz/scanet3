package org.scanet.core

import java.net.InetAddress
import java.nio.file.{Files, Paths}
import java.time.Instant

import com.google.protobuf.ByteString
import org.scanet.native.TfRecords
import org.tensorflow.util.Event

import scala.reflect.io.Path._

object TensorBoard {

  def write[A](ops: List[Output[_]], dir: String): Unit = {
    val graph = Session.using(_.toGraph(ops))
    val version = Event.newBuilder()
      .setFileVersion("brain.Event:2")
      .build()
    val graphDef = Event.newBuilder()
      .setGraphDef(ByteString.copyFrom(graph.toGraphDef))
      .build()
    dir.toDirectory.createDirectory()
    val computerName = InetAddress.getLocalHost.getHostName
    val file = s"events.out.tfevents.${Instant.now().getEpochSecond}.$computerName"
    TfRecords.of(List(version, graphDef))
      .writeTo(Files.newOutputStream(Paths.get(dir).resolve(file)))
  }
}
