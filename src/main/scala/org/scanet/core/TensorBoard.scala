package org.scanet.core

import java.net.InetAddress
import java.nio.file.{Files, Paths}
import java.time.Instant

import com.google.protobuf.ByteString
import org.scanet.math.Convertible
import org.scanet.native.TfRecords
import org.tensorflow.framework.Summary
import org.tensorflow.util.Event

import scala.reflect.io.Path._

class TensorBoard(val dir: String = "") {

  dir.toDirectory.createDirectory()

  def addGraph[A](ops: Output[_]*): Unit = {
    val graph = Session.using(_.toGraph(ops))
    val event = Event.newBuilder()
      .setGraphDef(ByteString.copyFrom(graph.toGraphDef))
      .build()
    writeEvents(event)
  }

  def addScalar[A](tag: String, value: A, step: Int)(implicit c: Convertible[A, Float]): Unit = {
    val summary = Summary.newBuilder().addValue(
      Summary.Value.newBuilder()
        .setTag(tag)
        .setSimpleValue(c.convert(value)).build())
    val event = Event.newBuilder()
      .setSummary(summary)
      .setStep(step)
      .build()
    writeEvents(event)
  }

  def writeEvents(events: Event*): Unit = {
    val version = Event.newBuilder()
      .setFileVersion("brain.Event:2")
      .build()
    val computerName = InetAddress.getLocalHost.getHostName
    val file = s"events.out.tfevents.${Instant.now().toEpochMilli}.$computerName"
    TfRecords.of(List(version) ++ events)
      .writeTo(Files.newOutputStream(Paths.get(dir).resolve(file)))
  }
}