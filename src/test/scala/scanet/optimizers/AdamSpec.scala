package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Tensor
import scanet.estimators.accuracy
import scanet.math.syntax._
import scanet.models.{LinearRegression, LogisticRegression}
import scanet.models.Loss._
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdamSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Adam" should "minimize linear regression" in {
    val ds = linearFunction
    val start = System.currentTimeMillis()
    val trained = ds
      .train(LinearRegression)
      .loss(MeanSquaredError)
      .using(Adam(rate = 0.1f))
      .initWith(Tensor.zeros(_))
      .batch(97)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    loss(x, y).toScalar should be <= 10f
    val stop = System.currentTimeMillis()
    println(s"took ${stop - start}")
  }

  it should "minimize logistic regression" in {
    val ds = logisticRegression.map(a => Array(a(0) / 100, a(1) / 100, a(2)))
    val trained = ds
      .train(LogisticRegression)
      .loss(BinaryCrossentropy)
      .using(Adam(0.1f))
      .initWith(s => Tensor.zeros(s))
      .batch(100)
      .each(1.epochs, RecordLoss())
      .stopAfter(100.epochs)
      .run()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 100).next()
    val loss = trained.loss.compile()
    loss(x, y).toScalar should be <= 0.4f
    accuracy(trained, ds) should be >= 0.9f
    val predictor = trained.result.compile()
    val input = Tensor.matrix(Array(0.3462f, 0.7802f), Array(0.6018f, 0.8630f))
    predictor(input).const.round.eval should be(Tensor.matrix(Array(0f), Array(1f)))
  }
}
