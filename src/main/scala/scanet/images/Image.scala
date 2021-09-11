package scanet.images

import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream

import javax.imageio.ImageIO
import scanet.core.Tensor
import scanet.core.syntax._

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

  /** Encode 3-D tensor as an image.
    *
    * A tensor should have 3 dimensions and the last dimension should:
    * - have appropriate size (1 for greyscale, 3 for RGB, 4 for RGBS color scheme)
    * - have float values in a range `[-1, 1]`
    *   where `-1` is absence of color (i.e. black for grayscale)
    *   and `1` is max value (i.e. white for grayscale)
    *
    * @param tensor 3-D tensor (height, width, channels)
    * @param channel color scheme
    * @param format image format (`png`, `jpeg`, `gif`, `bmp`)
    * @return encoded image
    */
  def encode(tensor: Tensor[Float], channel: Channel, format: String = "png"): Array[Byte] = {
    val dims = tensor.shape.dims
    val (height, width) = (dims(0), dims(1))
    val image = new BufferedImage(width, height, channel.code)
    val raster = image.getData.createCompatibleWritableRaster(width, height)
    for {
      x <- 0 until width
      y <- 0 until height
    } raster.setPixel(
      x,
      y,
      tensor(y, x).toArray.map(p => {
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
