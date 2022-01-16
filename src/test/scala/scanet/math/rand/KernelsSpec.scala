package scanet.math.rand

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Shape, Tensor}
import scanet.math.rand.Dist._
import scanet.syntax._

class KernelsSpec extends AnyWordSpec with Matchers {

  "random" should {

    "generate tensor with uniform distribution" in {
      val tensor = rand[Double](Shape(5), dist = Uniform, seed = Some(1)).roundAt(2)
      tensor.eval shouldBe Tensor.vector(0.5, 0.13, 0.5, 0.03, 0.36)
    }

    "generate tensor with standard normal distribution" in {
      val tensor = rand[Double](Shape(5), dist = Normal, seed = Some(4)).roundAt(2)
      tensor.eval shouldBe Tensor.vector(1.18, 0.52, 0.52, 0.71, -2.34)
    }

    "generate tensor with standard normal truncated to 2 std distribution" in {
      val tensor = rand[Double](Shape(5), dist = NormalTruncated, seed = Some(6)).roundAt(2)
      tensor.eval shouldBe Tensor.vector(-0.96, -0.16, 0.49, 0.79, -1.54)
    }
  }
}
