package scanet.models

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.math.syntax._
import scanet.models.Regularization.{L1, L2, Zero}
import scanet.test.CustomMatchers

class RegularizationSpec extends AnyWordSpec with CustomMatchers {

  "Zero regularization" should {
    "always return zero" in {
      Zero.build(Tensor.vector(-1f, 2f, 3f).const).eval should be(Tensor.scalar(0f))
    }
  }

  "L1 regularization" should {
    "compute sum of absolute weights divided by 2 with lambda coefficient" in {
      L1(lambda = 0.1f).build(Tensor.vector(-1f, 2f, 3f).const).eval should be(Tensor.scalar(0.3f))
    }
  }

  "L2 regularization" should {
    "compute sum of squared weights divided by 2 with lambda coefficient" in {
      L2(lambda = 0.1f).build(Tensor.vector(-1f, 2f, 3f).const).eval should be(Tensor.scalar(0.7f))
    }
  }
}
