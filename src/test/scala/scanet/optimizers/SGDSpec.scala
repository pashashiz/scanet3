package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Params.Weights
import scanet.core.{Params, Tensor}
import scanet.math.syntax._
import scanet.models.LinearRegression
import scanet.models.Loss._
import scanet.models.Math.`x^2`
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class SGDSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Plain SGD" should "minimize x^2" in {
    val trained = Optimizer
      .minimize[Float](`x^2`)
      .loss(Identity)
      .using(SGD(rate = 0.1f))
      .initParams(Params(Weights -> Tensor.scalar(5.0f)))
      .each(1.epochs, RecordLoss())
      .on(zero)
      .batch(1)
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    // x = weights
    val y = loss(Tensor.scalar(0.0f), Tensor.scalar(0.0f))
    y.toScalar should be <= 0.1f
  }

  "Plain SGD" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(SGD())
      .each(1.epochs, RecordLoss())
      .batch(97)
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    // note: that reaches 4.5 in 1500 epochs
    loss(x, y).toScalar should be <= 11f
  }

  "SGD with momentum" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(SGD(momentum = 0.1f))
      .batch(97)
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    // note: that reaches 4.5 in 1500 epochs
    loss(x, y).toScalar should be <= 11f
  }

  "SGD with Nesterov acceleration" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(SGD(momentum = 0.1f, nesterov = true))
      .batch(97)
      .stopAfter(100.epochs)
      .each(1.epochs, RecordLoss())
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    // note: that reaches 4.5 in 1500 epochs
    loss(x, y).toScalar should be <= 11f
  }
}
