package org.scanet.native

import java.nio.{ByteBuffer, ByteOrder}
import java.nio.ByteOrder.BIG_ENDIAN
import java.util.zip.CRC32

import com.google.common.hash.{Hashing => JavaHashing}

object Hashing {

  val mask: Long = 0xa282ead8L

  def mask(value: Int): Int = {
    val unsigned = unsignedInt(value)
    val masked = ((unsigned >>> 15) | (unsigned << 17)) + mask
    (masked & 0xFFFFFFFF).toInt
  }

  def unmask(value: Int): Int = {
    val unsigned = unsignedInt(value) - mask
    val unmasked = (unsigned >>> 17) | (unsigned << 15)
    (unmasked & 0xFFFFFFFF).toInt
  }

  def unsignedInt(int: Int): Long = int & 0x00000000ffffffffL

  def crc32(bytes: Array[Byte]): Int = {
    val coder = new CRC32()
    coder.update(bytes)
    (coder.getValue & 0xFFFFFFFF).toInt
  }

  def crc32c(bytes: Array[Byte]): Int = {
    JavaHashing.crc32c().hashBytes(bytes).asInt()
  }

  def crc32Masked(bytes: Array[Byte]): Int = mask(crc32(bytes))

  def crc32cMasked(bytes: Array[Byte]): Int = mask(crc32c(bytes))

  def toBytes(value: Long, order: ByteOrder = BIG_ENDIAN): Array[Byte] = {
    val buffer = ByteBuffer.allocate(8).order(order)
    buffer.putLong(value)
    buffer.array
  }
}
