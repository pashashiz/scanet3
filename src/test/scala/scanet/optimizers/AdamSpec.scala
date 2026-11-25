package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.{Params, Tensor}
import scanet.estimators.accuracy
import scanet.math.syntax._
import scanet.models.Initializer.Zeros
import scanet.models.Loss._
import scanet.models.layer.Dense
import scanet.models.{Activation, LinearRegression, LogisticRegression}
import scanet.optimizers.Adam.{Momentum, Velocity}
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdamSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Adam" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds
      .train(LinearRegression())
      .loss(MeanSquaredError)
      .using(Adam(rate = 0.1f))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    loss(x, y).toScalar should be <= 12f
  }

  it should "minimize one dense layer" in {
    val ds = linearFunction
    val trained = ds
      .train(Dense(1, Activation.Identity, biasInitializer = Zeros, kernelInitializer = Zeros))
      .loss(MeanSquaredError)
      .using(Adam(rate = 0.1f))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile
    val TRecord(x, y) = ds.firstTensor(97)
    loss(x, y).toScalar should be <= 12f
  }

  it should "minimize logistic regression" in {
    val ds = logisticRegression
    val trained = ds
      .train(LogisticRegression())
      .loss(BinaryCrossentropy)
      .using(Adam(0.1f))
      .batch(100)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val TRecord(x, y) = ds.firstTensor(100)
    val loss = trained.loss.compile
    loss(x, y).toScalar should be <= 0.4f
    accuracy(trained, ds) should be >= 0.9f
    val predictor = trained.result.compile
    val input = Tensor.matrix(Array(0.3462f, 0.7802f), Array(0.6018f, 0.8630f))
    predictor(input).const.round.eval should be(Tensor.matrix(Array(0f), Array(1f)))
  }

  "Adam" should "make the first step equal to the learning rate" in {
    // m = 0.1 * 2 = 0.2, v = 0.001 * 4 = 0.004, m^ = 2, v^ = 4
    // delta = rate * m^ / (sqrt(v^) + eps) = 0.001 * 2 / 2
    val zero = Tensor.scalar(0f).const
    val Delta(delta, next) = Adam(rate = 0.001f)
      .build[Float](2f.const, Params(Momentum -> zero, Velocity -> zero), 1.const)
    delta.eval should beCloseTo(Tensor.scalar(0.001f))
    next(Momentum).eval should beCloseTo(Tensor.scalar(0.2f))
    next(Velocity).eval should beCloseTo(Tensor.scalar(0.004f))
  }

  "Adam" should "add epsilon after the square root" in {
    // with a tiny gradient m^ = 1e-4 and sqrt(v^) = 1e-4 so the step is still ~ rate,
    // with epsilon inside the square root it would be sqrt(1e-8 + eps), ~3x smaller for eps = 1e-7
    val zero = Tensor.scalar(0f).const
    val Delta(delta, _) = Adam(rate = 0.001f, epsilon = 1e-7f)
      .build[Float](1e-4f.const, Params(Momentum -> zero, Velocity -> zero), 1.const)
    delta.eval should beCloseTo(Tensor.scalar(0.000999f))
  }
}
