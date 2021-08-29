package org.scanet.core

import java.lang.System.currentTimeMillis
import java.net.InetAddress
import java.nio.file.{Files, Paths}
import java.time.Instant

import com.google.protobuf.ByteString
import org.scanet.images.{Channel, Image}
import org.scanet.math.Convertible
import org.scanet.native.TfRecords
import org.tensorflow.proto.framework.Summary
import org.tensorflow.proto.util.Event

import scala.reflect.io.Path._

case class TensorBoard(dir: String = "") {

  dir.toDirectory.createDirectory()

  def addGraph[A](ops: Expr[_]*): TensorBoard = {
    val graph = Session.withing(_.toGraph(ops))
    val event =
      Event.newBuilder().setGraphDef(ByteString.copyFrom(graph.toGraphDef.toByteArray)).build()
    writeEvents(event)
  }

  def addScalar[A](tag: String, value: A, step: Int)(
      implicit c: Convertible[A, Float]): TensorBoard = {
    val summary = Summary
      .newBuilder()
      .addValue(Summary.Value.newBuilder().setTag(tag).setSimpleValue(c.convert(value)).build())
    val event = Event
      .newBuilder()
      .setSummary(summary)
      .setWallTime(currentTimeMillis() / 1000)
      .setStep(step)
      .build()
    writeEvents(event)
  }

  def addImage(tag: String, value: Tensor[Float], channel: Channel): TensorBoard = {
    val dims = value.shape.dims
    val image = Summary.Image
      .newBuilder()
      .setHeight(dims(0))
      .setWidth(dims(1))
      .setColorspace(dims(2))
      .setEncodedImageString(ByteString.copyFrom(Image.encode(value, channel)))
      .build()
    val summary = Summary
      .newBuilder()
      .addValue(Summary.Value.newBuilder().setTag(tag).setImage(image).build())
      .build()
    val event = Event.newBuilder().setSummary(summary).build()
    writeEvents(event)
  }

  def writeEvents(events: Event*): TensorBoard = {
    val version = Event.newBuilder().setFileVersion("brain.Event:2").build()
    val computerName = InetAddress.getLocalHost.getHostName
    val file = s"events.out.tfevents.${Instant.now().toEpochMilli}.$computerName"
    TfRecords
      .of(List(version) ++ events)
      .writeTo(Files.newOutputStream(Paths.get(dir).resolve(file)))
    this
  }
}
