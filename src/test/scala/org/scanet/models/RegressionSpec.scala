package org.scanet.models

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.datasets.CSVDataset
import org.scanet.math.syntax._
import org.scanet.optimizers.Effect._
import org.scanet.optimizers.syntax._
import org.scanet.optimizers.{AdaDelta, AdaGrad, Optimizer, SGD}
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

  it should "be minimized" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaGrad())
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .each(logResult())
      //.each(100.epochs, plotResult())
      .stopAfter(1500.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be(4.64f +- 0.01f)
  }

  "facebook comments model" should "predict number of comments" ignore {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaGrad(rate = 0.5))
      .on(ds)
      .batch(1000)
      .each(1.iterations, logResult())
      // .each(1.epochs, plotResult())
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val err = regression(ds.iterator.next(100), weights)
    println(err)
  }
}
