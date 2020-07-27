package org.scanet

import java.io.{ByteArrayOutputStream, InputStream}


package object test {

  def resourceAsStream(path: String): InputStream =
    getClass.getClassLoader.getResourceAsStream(path)

  def resourceAsBytes(path: String): Array[Byte] = {
    val is = resourceAsStream(path)
    val buffer = new ByteArrayOutputStream()
    val data = new Array[Byte](1024)
    var nRead = 0
    while ({
      nRead = is.read(data, 0, data.length)
      nRead != -1
    }) {
      buffer.write(data, 0, nRead)
    }
    buffer.flush()
    buffer.toByteArray
  }
}
