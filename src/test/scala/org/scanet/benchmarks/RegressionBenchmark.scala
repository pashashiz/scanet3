package org.scanet.benchmarks

import org.scalatest.Ignore
import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.datasets.CSVDataset
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.Effect.{logResult, plotResult}
import org.scanet.optimizers.syntax._
import org.scanet.optimizers.{AdaDelta, AdaGrad, Adam, Optimizer, RMSProp, SGD}
import org.scanet.test.CustomMatchers

@Ignore
class RegressionBenchmark extends AnyFlatSpec with CustomMatchers {

  "linear regression" should "be minimized by plain SDG" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/SDG"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 1f
  }

  it should "be minimized by SDG with momentum" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.9))
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Momentum"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 1f
  }

  it should "be minimized by SDG with nesterov acceleration" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.9, nesterov = true))
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Nesterov"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 1f
  }

  it should "be minimized by AdagGrad" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaGrad())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/AdagGrad"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 0.5f
  }

  it should "be minimized by AdagDelta" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaDelta())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/AdagDelta"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 0.5f
  }

  it should "be minimized by RMSProp" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(RMSProp())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/RMSProp"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 0.5f
  }

  it should "be minimized by Adam" in {
    val ds = CSVDataset("facebook-comments-scaled.csv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(Adam(rate = 0.1))
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Adam"))
      .stopAfter(10.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(1000), weights)
    result.toScalar should be <= 0.5f
  }
}
