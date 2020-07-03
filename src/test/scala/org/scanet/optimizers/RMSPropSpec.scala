package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.{LinearRegression, MeanSquaredError}
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class RMSPropSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "RMSProp" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = Optimizer
      .minimize[Float](LinearRegression)
      .loss(MeanSquaredError)
      .using(RMSProp(rate = 0.06f))
      .initWith(Tensor.zeros(_))
      .on(ds)
      .batch(97)
      .each(Condition.always, Effect.logResult())
      .stopAfter(100.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    loss(x, y).toScalar should be <= 9.4f
  }
}
