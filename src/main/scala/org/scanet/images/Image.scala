package org.scanet.images

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream

import javax.imageio.ImageIO
import org.scanet.core.Tensor
import org.scanet.core.syntax._

sealed trait Channel {
  def code: Int
}

case class RGB() extends Channel {
  override def code = 1
}

case class RGBA() extends Channel {
  override def code = 2
}

case class Grayscale() extends Channel {
  override def code = 10
}

object Image {

  def encode(tensor: Tensor[Float], channel: Channel, format: String = "png"): Array[Byte] = {
    val dims = tensor.shape.dims
    val (height, width) = (dims(0), dims(1))
    val image = new BufferedImage(width, height, channel.code)
    val raster = image.getData.createCompatibleWritableRaster(width, height)
    for {
      x <- 0 until width
      y <- 0 until height
    } raster.setPixel(x, y, tensor(y, x).toArray.map(p => {
      val pp = math.max(0f, math.min((p + 1f) / 2f, 1f))
      (pp * 255).toInt
    }))
    image.setData(raster)
    val out = new ByteArrayOutputStream()
    ImageIO.write(image, format, out)
    out.toByteArray
  }

  def decode(image: Array[Byte], channel: Channel, format: String = "png"): Tensor[Float] = ???
}
