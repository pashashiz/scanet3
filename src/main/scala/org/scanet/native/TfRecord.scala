package org.scanet.native

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder.LITTLE_ENDIAN

import com.google.protobuf.MessageLite
import org.scanet.native.Hashing.crc32cMasked

/**
 * Format of a single record:
 * uint64    length
 * uint32    masked crc32c of length
 * byte      data[length]
 * uint32    masked crc32c of data
 */
case class TfRecord(message: MessageLite) {

  def toBytes: Array[Byte] = {
    val messageBytes = message.toByteArray
    ByteBuffer
      .allocate(messageBytes.size + 16)
      .order(LITTLE_ENDIAN)
      .putLong(messageBytes.size)
      .putInt(crc32cMasked(Hashing.toBytes(messageBytes.size, order = LITTLE_ENDIAN)))
      .put(messageBytes)
      .putInt(crc32cMasked(messageBytes))
      .array()
  }

  def writeTo(out: OutputStream): OutputStream = {
    out.write(toBytes)
    out
  }
}

case class TfRecords(records: Seq[TfRecord]) {
  def writeTo(out: OutputStream): OutputStream =
    records.foldLeft(out)((acc, next) => next.writeTo(acc))
}

object TfRecords {
  def of(records: Seq[MessageLite]): TfRecords = new TfRecords(records.map(TfRecord))
}