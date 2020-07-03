package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.{LinearRegression, MeanSquaredError}
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdaDeltaSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "AdaDelta" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(AdaDelta())
      .initWith(Tensor.zeros(_))
      .batch(97)
      .each(1.epochs, logResult())
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    // note: that reaches 4.5 in 2000 epochs
    loss(x, y).toScalar should be <= 50f
  }
}
