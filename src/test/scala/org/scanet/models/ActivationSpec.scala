package org.scanet.models

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._

class ActivationSpec extends AnyFlatSpec with CustomMatchers {

  "softmax" should "return probabilities which are equal to 1 when summed" in {
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
