package scanet.models

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.math.syntax._
import scanet.models.Activation._
import scanet.test.CustomMatchers

class ActivationSpec extends AnyWordSpec with CustomMatchers {

  "tanh activation function" should {

    "be like regular tanh function" in {
      Tanh.build(0.5f.const).eval should be(Tensor.scalar(0.46211717f))
    }
  }

  "softmax activation function" should {

    "return probabilities which are equal to 1 when summed" in {
      val inputs = Tensor.matrix(
        Array(1.3f, 5.1f, 2.2f, 0.7f, 1.1f),
        Array(2.1f, 2.2f, 0.1f, 3.2f, 1.1f)
      )
      val expected = Tensor.matrix(
        Array(0.020190466f, 0.9025377f, 0.049660537f, 0.011080763f, 0.016530557f),
        Array(0.17817205f, 0.19691059f, 0.024112968f, 0.53525853f, 0.06554584f)
      )
      Softmax.build(inputs.const).eval should be(expected)
    }
  }

  "ReLU activation function" should {

    "be like identity function when > 0" in {
      ReLU().build(5f.const).eval should be(Tensor.scalar(5f))
    }

    "return 0 when input is 0" in {
      ReLU().build(0f.const).eval should be(Tensor.scalar(0f))
    }

    "return 0 when input is < 0" in {
      ReLU().build((-5f).const).eval should be(Tensor.scalar(0f))
    }
  }
}
