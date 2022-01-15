package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.syntax._
import scanet.test.CustomMatchers

import scala.collection.immutable.Seq

class Pool2DLayerSpec extends AnyWordSpec with CustomMatchers {

  "Pool2D layer" should {

    "pool input signal with a given window" in {
      val model = Pool2D(window = (2, 2))
      val input = Tensor.matrix[Double](
        Array(2, 1, 2, 0, 1),
        Array(1, 3, 2, 2, 3),
        Array(1, 1, 3, 3, 0),
        Array(2, 2, 0, 1, 1),
        Array(0, 0, 3, 1, 2))
        .reshape(1, 5, 5, 1)
        .compact
      val output = Tensor.matrix[Double](
        Array(3.0, 3.0, 2.0, 3.0),
        Array(3.0, 3.0, 3.0, 3.0),
        Array(2.0, 3.0, 3.0, 3.0),
        Array(2.0, 3.0, 3.0, 2.0))
        .reshape(1, 4, 4, 1)
      val result = model.result[Double].compile()
      result(input, Seq.empty).const.roundAt(2).eval shouldBe output
    }

    "have string repr" in {
      val model = Pool2D(window = (2, 2))
      model.toString shouldBe "Pool2D((2,2),(1,1),Valid,NHWC,Max)"
    }
  }
}
