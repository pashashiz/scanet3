package org.scanet.benchmarks

import org.scalatest.Ignore
import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.math.syntax._
import org.scanet.models.{LinearRegression, MeanSquaredError}
import org.scanet.optimizers.Effect.RecordLoss
import org.scanet.optimizers.syntax._
import org.scanet.optimizers._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

@Ignore
class RegressionBenchmark extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "linear regression" should "be minimized by plain SDG" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(SGD())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/SDG"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by SDG with momentum" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(SGD(momentum = 0.9f))
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/Momentum"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by SDG with nesterov acceleration" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(SGD(momentum = 0.9f, nesterov = true))
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/Nesterov"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by AdagGrad" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(AdaGrad())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/AdaGrad"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by AdagDelta" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(AdaDelta())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/AdaDelta"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by RMSProp" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(RMSProp())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/RMSProp"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by Adam" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(Adam(rate = 0.1f))
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/Adam"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by Adamax" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(Adamax())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/Adamax"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by Nadam" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(Nadam())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/Nadam"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }

  it should "be minimized by AMSGrad" in {
    val ds = facebookComments
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(AMSGrad())
      .batch(1000)
      .each(1.epochs, RecordLoss(tensorboard = true, dir = "board/AMSGrad"))
      .stopAfter(10.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
    loss(x, y).toScalar should be <= 1f
  }
}
