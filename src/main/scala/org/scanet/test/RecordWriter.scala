package org.scanet.test

import java.nio.ByteBuffer
import java.nio.ByteOrder.LITTLE_ENDIAN
import java.nio.file.{Files, Paths}
import org.tensorflow.util.Event

import org.scanet.native.Hashing
import org.scanet.native.Hashing.crc32cMasked

object RecordWriter {
  // Format of a single record:
  //  uint64    length
  //  uint32    masked crc of length
  //  byte      data[length]
  //  uint32    masked crc of data
  def main(args: Array[String]): Unit = {
//    val bytes = Files.readAllBytes(Paths.get("sample/events.out.tfevents.1583612234.Pavlos-MacBook-Pro-2.local"))
    val bytes = Files.readAllBytes(Paths.get("events.out.tfevents.1583612234.Pavlos-MacBook-Pro-2.local"))
    val buffer = ByteBuffer.wrap(bytes).order(LITTLE_ENDIAN)
    while (buffer.hasRemaining) {
      val length = buffer.getLong
      println(s"reading next: $length bytes")
      val lengthCRC32 = buffer.getInt
      println(s"original length CRC32: $lengthCRC32")
      println(s"calculated length CRC32: ${crc32cMasked(Hashing.toBytes(length, order = LITTLE_ENDIAN))}")
      val data = Array.ofDim[Byte](length.toInt)
      buffer.get(data)
      println(new String(data, "UTF-8"))
      val dataCRC32 = buffer.getInt
      println(s"original data CRC32: $dataCRC32")
      println(s"calculated data CRC32: ${crc32cMasked(data)}")
    }
  }
}
