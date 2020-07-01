package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.estimators.accuracy
import org.scanet.math.syntax._
import org.scanet.models.{LinearRegression, LogisticRegression}
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdamSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Adam" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = Optimizer
      .minimize(LinearRegression)
      .using(Adam(rate = 0.1f))
      .initWith(Tensor.zeros(_))
      .on(ds)
      .batch(97)
      .each(1.epochs, logResult())
      .stopAfter(100.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    loss(x, y).toScalar should be <= 4.9f
  }

  it should "minimize logistic regression" in {
    val ds = logisticRegression.map(a => Array(a(0)/100, a(1)/100, a(2)))
    val trained = Optimizer
      .minimize(LogisticRegression)
      .using(Adam(0.1f))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(100)
      .each(1.epochs, logResult())
      .stopAfter(100.epochs)
      .build
      .run()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 100).next()
    val loss = trained.loss.compile()
    loss(x, y).toScalar should be <= 0.4f
    accuracy(trained, ds) should be >= 0.9f
    val predictor = trained.result.compile()
    val input = Tensor.matrix(
      Array(0.3462f, 0.7802f),
      Array(0.6018f, 0.8630f),
    )
    predictor(input).const.round.eval should be(Tensor.matrix(Array(0f), Array(1f)))
  }
}
