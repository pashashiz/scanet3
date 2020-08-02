package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.LinearRegression
import org.scanet.models.Loss._
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AdaGradSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "AdaGrad" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = ds.train(LinearRegression)
      .loss(MeanSquaredError)
      .using(AdaGrad())
      .initWith(Tensor.zeros(_))
      .batch(97)
      .stopAfter(100.epochs)
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    loss(x, y).toScalar should be <= 9f
  }
}
