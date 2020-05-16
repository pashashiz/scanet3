package org.scanet.models

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.datasets.CSVDataset
import org.scanet.math.syntax._
import org.scanet.optimizers.{Optimizer, SGD}
import org.scanet.test.CustomMatchers

class RegressionSpec extends AnyFlatSpec with CustomMatchers {

  "linear regression" should "calculate approximation error " in {
    val reg = Regression.linear
    val x = Tensor.matrix(
      Array(1.0f, 2.0f, 6.0f),
      Array(2.0f, 4.0f, 12.0f)).const
    val weights = Tensor.vector(1.0f, 2.0f, 3.0f).const
    reg(x, weights).eval should be(Tensor.scalar(8.5f))
  }

  it should "be minimized" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.9, nesterov = true))
      .batch(97)
      .initWith(Tensor.zeros(2))
      .on(ds)
      .epochs(1500)
      .doOnEach(step => {
        println(s"${step.iter}")
      })
      .build
      .run()
    val result = Regression.linear(ds.iterator.next(97).const, weights.const).eval
    result.toScalar should be(4.56f +- 0.01f)
  }
}
