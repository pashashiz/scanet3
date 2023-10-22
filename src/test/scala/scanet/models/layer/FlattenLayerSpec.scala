package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Params, Tensor}
import scanet.syntax._
import scanet.test.CustomMatchers

class FlattenLayerSpec extends AnyWordSpec with CustomMatchers {

  "Flatten layer" should {
    "flatten the output from (batch, features_1, ... features_n) into (batch, features) tensor" in {
      val x = Tensor.ones[Float](10, 5, 5)
      val model = Flatten
      val result = model.result[Float].compile
      result(x, Params.empty) shouldBe x.reshape(10, 25)
    }
  }
}
