package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Params, Tensor}
import scanet.models.Activation.Sigmoid
import scanet.syntax._
import scanet.test.CustomMatchers

import scala.collection.immutable.Seq

class ActivateLayerSpec extends AnyWordSpec with CustomMatchers {

  "activate layer" should {

    "apply activation function to the input" in {
      val x = Tensor.matrix(
        Array(0f, 0f, 1f),
        Array(0f, 1f, 1f),
        Array(1f, 0f, 1f),
        Array(1f, 1f, 1f))
      val model = Sigmoid.layer
      val result = model.result[Float].compile
      val y = Tensor.matrix(
        Array(0.5f, 0.5f, 0.7310586f),
        Array(0.5f, 0.7310586f, 0.7310586f),
        Array(0.7310586f, 0.5f, 0.7310586f),
        Array(0.7310586f, 0.7310586f, 0.7310586f))
      result(x, Params.empty) should be(y)
    }

    "have string repr" in {
      Sigmoid.layer.toString shouldBe "Sigmoid"
    }
  }
}
