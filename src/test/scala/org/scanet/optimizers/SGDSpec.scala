package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.datasets.{CSVDataset, EmptyDataset}
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.syntax._
import org.scanet.test.CustomMatchers
import org.scanet.models.Math._

class SGDSpec extends AnyFlatSpec with CustomMatchers {

  "Plain SGD" should "minimize x^2" in {
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SGD(rate = 0.1))
      .initWith(Tensor.scalar(5.0f))
      .on(EmptyDataset())
      .epochs(20)
      .build
    val x = opt.run()
    val y = `x^2`.result.compile()(Tensor.scalar(0.0f), x)
    y.toScalar should be <= 0.1f
  }

  "Plain SGD" should "minimize linear regression" in {
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

  "SGD with momentum" should "minimize linear regression" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.1f))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .stopAfter(1300.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }

  "SGD with Nesterov acceleration" should "minimize linear regression" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.1f, nesterov = true))
      .initWith(Tensor.zeros(_))
      .on(ds)
      .stopAfter(1300.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }
}
