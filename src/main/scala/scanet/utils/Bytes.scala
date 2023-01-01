package scanet.utils

object Bytes {
  def formatSize(bytes: Long): String = {
    if (bytes < 1024) return s"$bytes B"
    val z = (63 - java.lang.Long.numberOfLeadingZeros(bytes)) / 10
    "%.1f %sB".format(bytes.toDouble / (1L << (z * 10)), " KMGTPE".charAt(z))
  }
}
