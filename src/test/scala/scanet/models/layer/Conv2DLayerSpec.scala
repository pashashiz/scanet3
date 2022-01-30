package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.syntax._
import scanet.test.CustomMatchers

import scala.collection.immutable.Seq

class Conv2DLayerSpec extends AnyWordSpec with CustomMatchers {

  "Conv2D layer" should {

    "convolve input signal with a given kernel" in {
      val model = Conv2D(filters = 1, kernel = (2, 2))
      val input = Tensor.matrix[Double](
        Array(2, 1, 2, 0, 1),
        Array(1, 3, 2, 2, 3),
        Array(1, 1, 3, 3, 0),
        Array(2, 2, 0, 1, 1),
        Array(0, 0, 3, 1, 2))
        .reshape(1, 5, 5, 1)
        .compact
      val filters = Tensor.matrix[Double](
        Array(2, 3),
        Array(0, 1))
        .reshape(2, 2, 1, 1)
        .compact
      val output = Tensor.matrix[Double](
        Array(10.0, 10.0, 6.0, 6.0),
        Array(12.0, 15.0, 13.0, 13.0),
        Array(7.0, 11.0, 16.0, 7.0),
        Array(10.0, 7.0, 4.0, 7.0))
        .reshape(1, 4, 4, 1)
      val result = model.result[Double].compile
      result(input, Seq(filters)).const.roundAt(2).eval shouldBe output
    }

    "have string repr" in {
      val model = Conv2D(filters = 1, kernel = (2, 2)).toString
      model shouldBe "Conv2D(1,(2,2),(1,1),Valid,NHWC,GlorotUniform(None))"
    }
  }
}
