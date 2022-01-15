package scanet.models

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Shape, Tensor}
import scanet.math.syntax._
import scanet.models.Initializer._
import scanet.test.CustomMatchers

class InitializerSpec extends AnyWordSpec with CustomMatchers {

  "zeros initializer" should {
    "initialize a tensor with zeros" in {
      Zeros.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(0f, 0f, 0f)
    }
  }

  "ones initializer" should {
    "initialize a tensor with zeros" in {
      Ones.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(1f, 1f, 1f)
    }
  }

  "random uniform initializer" should {
    "initialize a tensor withing given range" in {
      val rn = RandomUniform(min = -1, max = 1, seed = Some(1))
      rn.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(0.63f, -0.84f, 0.78f)
    }
  }

  "random normal initializer" should {
    "initialize a tensor using normal distribution given mean and std" in {
      val rn = RandomNormal(std = 1, seed = Some(1))
      rn.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(0.31f, 0.57f, 0.42f)
    }
  }

  "random normal truncated initializer" should {
    "initialize a tensor using normal distribution given mean and std with range [-2 * std; +2 * std]" in {
      val rn = RandomNormalTruncated(std = 1, seed = Some(1))
      rn.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(0.31f, 0.57f, 0.42f)
    }
  }

  "Glorot (Xavier) uniform initializer" should {
    "initialize a tensor using uniform distribution and scale based on number of weights" in {
      val rn = GlorotUniform(seed = Some(1))
      rn.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(0.63f, -0.84f, 0.78f)
    }
  }

  "Glorot (Xavier) normal initializer" should {
    "initialize a tensor using normal truncated distribution and scale based on number of weights" in {
      val rn = GlorotNormal(seed = Some(2))
      rn.build[Float](Shape(3)).roundAt(2).eval shouldBe Tensor.vector(-0.25f, -0.31f, -0.34f)
    }
  }
}
