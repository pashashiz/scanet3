package scanet.benchmarks

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Shape
import scanet.math.syntax._
import scanet.models.LinearRegression
import scanet.models.Loss.MeanSquaredError
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers._
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

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
      val (x, y) = TensorIterator(ds.collect.iterator, (Shape(53), Shape(1)), 1000).next()
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
