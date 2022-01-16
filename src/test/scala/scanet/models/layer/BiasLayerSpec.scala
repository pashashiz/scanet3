package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.syntax._
import scanet.test.CustomMatchers

import scala.collection.immutable.Seq

class BiasLayerSpec extends AnyWordSpec with CustomMatchers {

  "bias layer" should {

    "sum up a bias vector with last dimension" in {
      val x = Tensor.matrix(
        Array(0f, 0f, 1f),
        Array(0f, 1f, 1f),
        Array(1f, 0f, 1f),
        Array(1f, 1f, 1f))
      val b = Tensor.vector(1f, 2f, 3f)
      val model = Bias(3)
      val result = model.result[Float].compile()
      val y = Tensor.matrix(
        Array(1f, 2f, 4f),
        Array(1f, 3f, 4f),
        Array(2f, 2f, 4f),
        Array(2f, 3f, 4f))
      result(x, Seq(b)) should be(y)
    }

    "have string repr" in {
      Bias(3).toString shouldBe "Bias(3,Zero,Zeros)"
    }
  }
}
