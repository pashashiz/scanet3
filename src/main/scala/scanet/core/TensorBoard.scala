package scanet.core

import java.lang.System.currentTimeMillis
import java.net.InetAddress
import java.nio.file.{Files, Paths}
import java.time.Instant
import com.google.protobuf.ByteString
import scanet.images.{Channel, Image}
import scanet.native.TfRecords
import org.tensorflow.proto.framework.Summary
import org.tensorflow.proto.util.Event

import scala.reflect.io.Path._

case class TensorBoard(dir: String = "") {

  dir.toDirectory.createDirectory()

  def addGraph[A](ops: Expr[_]*): TensorBoard = {
    val graph = Session.withing(_.toGraph(ops.toList))
    val event =
      Event.newBuilder().setGraphDef(ByteString.copyFrom(graph.toGraphDef.toByteArray)).build()
    writeEvents(Seq(event))
  }

  def addScalar[A](tag: String, value: A, step: Int, path: Option[String] = None)(
      implicit c: Convertible[A, Float]): TensorBoard = {
    val summary = Summary
      .newBuilder()
      .addValue(Summary.Value.newBuilder().setTag(tag).setSimpleValue(c.convert(value)).build())
    val event = Event
      .newBuilder()
      .setSummary(summary)
      .setWallTime(currentTimeMillis().toDouble / 1000)
      .setStep(step)
      .build()
    writeEvents(Seq(event), path)
  }

  def addImage(tag: String, value: Tensor[Float], channel: Channel, path: Option[String] = None): TensorBoard = {
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
    writeEvents(Seq(event), path)
  }

  def writeEvents(events: Iterable[Event], subpath: Option[String] = None): TensorBoard = {
    val version = Event.newBuilder().setFileVersion("brain.Event:2").build()
    val computerName = InetAddress.getLocalHost.getHostName
    val file = s"events.out.tfevents.${Instant.now().toEpochMilli}.$computerName"
    val root = Paths.get(dir)
    val path = subpath.map(sb => root.resolve(sb)).getOrElse(root)
    TfRecords
      .of(List(version) ++ events)
      .writeTo(Files.newOutputStream(path.resolve(file)))
    this
  }
}
