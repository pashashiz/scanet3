package org.scanet.benchmarks

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import org.scanet.math.syntax._
import org.scanet.models.LinearRegression
import org.scanet.models.Loss.MeanSquaredError
import org.scanet.optimizers.Effect.RecordLoss
import org.scanet.optimizers._
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

trait RegressionBehaviours {

  this: AnyWordSpec with CustomMatchers with SharedSpark with Datasets =>

  def successfullyTrainedWith(alg: Algorithm): Unit = {
    "be successfully trained" in {
      val ds = facebookComments
      val trained = ds
        .train(LinearRegression)
        .loss(MeanSquaredError)
        .using(SGD())
        .batch(1000)
        .each(1.epochs, RecordLoss(tensorboard = true, dir = s"board/$alg"))
        .stopAfter(10.epochs)
        .run()
      val loss = trained.loss.compile()
      val (x, y) = Tensor2Iterator(ds.collect.iterator, 1000).next()
      loss(x, y).toScalar should be <= 1f
    }
  }
}

@Slow
class RegressionBenchmark
    extends AnyWordSpec
    with CustomMatchers
    with SharedSpark
    with Datasets
    with RegressionBehaviours {

  "Linear regression should be minimized" when {

    "used SDG algorithm" should {
      behave like successfullyTrainedWith(SGD())
    }

    "used SDG algorithm with momentum" should {
      behave like successfullyTrainedWith(SGD(momentum = 0.9f))
    }

    "used SDG algorithm with nesterov acceleration" should {
      behave like successfullyTrainedWith(SGD(momentum = 0.9f, nesterov = true))
    }

    "used AdaDelta algorithm" should {
      behave like successfullyTrainedWith(AdaDelta())
    }

    "used RMSProp algorithm" should {
      behave like successfullyTrainedWith(RMSProp())
    }

    "used Adam algorithm" should {
      behave like successfullyTrainedWith(Adam(0.1f))
    }

    "used AMSGrad algorithm" should {
      behave like successfullyTrainedWith(AMSGrad())
    }

    "used AdaGrad algorithm" should {
      behave like successfullyTrainedWith(AdaGrad(rate = 1f))
    }

    "used Adamax algorithm" should {
      behave like successfullyTrainedWith(Adamax())
    }

    "used Nadam algorithm" should {
      behave like successfullyTrainedWith(Nadam())
    }
  }
}
