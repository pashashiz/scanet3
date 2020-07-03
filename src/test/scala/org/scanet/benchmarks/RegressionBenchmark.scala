package org.scanet.benchmarks

import org.scalatest.Ignore
import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.math.syntax._
import org.scanet.models.{LinearRegression, MeanSquaredError}
import org.scanet.optimizers.Effect.{logResult, plotResult}
import org.scanet.optimizers.syntax._
import org.scanet.optimizers._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

@Ignore
class RegressionBenchmark extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "linear regression" should "be minimized by plain SDG" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(SGD())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/SDG"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by SDG with momentum" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(SGD(momentum = 0.9f))
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Momentum"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by SDG with nesterov acceleration" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(SGD(momentum = 0.9f, nesterov = true))
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Nesterov"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by AdagGrad" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(AdaGrad())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/AdaGrad"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by AdagDelta" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(AdaDelta())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/AdaDelta"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by RMSProp" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(RMSProp())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/RMSProp"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by Adam" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(Adam(rate = 0.1f))
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Adam"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by Adamax" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(Adamax())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Adamax"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by Nadam" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(Nadam())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/Nadam"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by AMSGrad" in {
    val ds = facebookComments
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(AMSGrad())
      .on(ds)
      .batch(1000)
      .each(1.epochs, logResult())
      .each(1.iterations, plotResult(name = "Error", dir = "board/AMSGrad"))
      .stopAfter(10.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }
}
