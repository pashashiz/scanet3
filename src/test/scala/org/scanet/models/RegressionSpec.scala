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

  it should "be minimized by plain SDG withing 1500 epoch" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD())
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .stopAfter(1500.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }

  it should "be minimized by SDG with momentum withing 1300 epoch" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.1))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .stopAfter(1300.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }

  it should "be minimized by SDG with nesterov momentum withing 1300 epoch" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.1, nesterov = true))
      .initWith(Tensor.zeros(_))
      .on(ds)
      .stopAfter(1300.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }

  it should "be minimized by Adagrad withing 100 epoch" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaGrad())
      .initWith(Tensor.zeros(_))
      .on(ds)
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }

  it should "be minimized by Adadelta withing 2000 epochs" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaDelta())
      .initWith(Tensor.zeros(_))
      .on(ds)
      .stopAfter(2000.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
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
