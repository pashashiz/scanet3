package scanet.images

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.core.Tensor
import scanet.core.syntax._
import scanet.test._

class ImageSpec extends AnyFlatSpec with Matchers {

  // note, encoding will not fully match on different processors due to different precision
  "3-D tensor" should "be encoded as png" ignore {
    val data = Tensor.matrix(Array(-1f, 1f, 1f), Array(0f, -0.3f, 0.3f)).reshape(2, 3, 1)
    val bytes = Image.encode(data, Grayscale())
    bytes should be(resourceAsBytes("tensor-image.png"))
  }
}
