package scanet.core

import org.tensorflow.internal.c_api.global.tensorflow.{
  TF_DeleteDeviceList,
  TF_DeviceListCount,
  TF_DeviceListName,
  TF_SessionListDevices
}
import scanet.native.attempt
import scala.collection.immutable.Seq

case class Device(name: String, tpe: String, index: Int)
object Device {
  def parse(name: String): Device = {
    val sections = name.split("/")
    require(sections.nonEmpty, s"device representation is invalid $name")
    val parts = sections.last.split(":")
    require(parts.size == 3, s"device representation is invalid $name")
    Device(name, parts(1), parts(2).toInt)
  }
}

object Devices {
  def list(session: Session): Seq[Device] = {
    val devices = attempt(TF_SessionListDevices(session.nativeHandle, _))
    try {
      val count = TF_DeviceListCount(devices)
      (0 until count)
        .map { index =>
          val name = attempt(TF_DeviceListName(devices, index, _)).getString
          Device.parse(name)
        }
        .toList
    } finally {
      TF_DeleteDeviceList(devices)
    }
  }
}
