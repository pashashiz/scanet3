package org.scanet.models

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

class RegressionSpec extends AnyFlatSpec with CustomMatchers {

  "linear regression" should "calculate approximation error " in {
    val regression = Regression.linear.result.compile()
    val x = Tensor.matrix(
      Array(1.0f, 2.0f, 6.0f),
      Array(2.0f, 4.0f, 12.0f))
    val weights = Tensor.vector(1.0f, 2.0f, 3.0f)
    regression(x, weights) should be(Tensor.scalar(8.5f))
  }
}
