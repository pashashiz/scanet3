package scanet.benchmarks

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import scanet.estimators.accuracy
import scanet.math.syntax._
import scanet.models.Activation.{Sigmoid, Softmax}
import scanet.models.Loss.CategoricalCrossentropy
import scanet.models.layer.{Dense, Flatten}
import scanet.optimizers.Effect.{RecordAccuracy, RecordLoss}
import scanet.optimizers._
import scanet.optimizers.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

trait ANN_MNISTBehaviours {

  this: AnyWordSpec with CustomMatchers with SharedSpark with Datasets =>

  def successfullyTrainedWith(alg: Algorithm): Unit = {
    "be successfully trained" in {
      val (trainingDs, testDs) = MNIST()
      val model = Flatten >> Dense(50, Sigmoid) >> Dense(10, Softmax)
      val trained = trainingDs
        .train(model)
        .loss(CategoricalCrossentropy)
        .using(alg)
        .batch(1000)
        .each(1.epochs, RecordLoss(tensorboard = true))
        .each(5.epochs, RecordAccuracy(testDs, tensorboard = true))
        .stopAfter(50.epochs)
        .board(s"board/$alg")
        .run()
      accuracy(trained, testDs) should be >= 0.9f
    }
  }
}

@Slow
class ANN_MNISTBenchmark
    extends AnyWordSpec
    with CustomMatchers
    with SharedSpark
    with Datasets
    with ANN_MNISTBehaviours {

  "MNIST dataset with Fully Connected Neural Network architecture" when {

    "used SDG algorithm" should {
      behave like successfullyTrainedWith(SGD(rate = 0.6f, momentum = 0.9f))
    }

    "used AdaDelta algorithm" should {
      behave like successfullyTrainedWith(AdaDelta())
    }

    "used RMSProp algorithm" should {
      behave like successfullyTrainedWith(RMSProp())
    }

    "used Adam algorithm" should {
      behave like successfullyTrainedWith(Adam(0.01f))
    }

    "used AMSGrad algorithm" should {
      behave like successfullyTrainedWith(AMSGrad())
    }

    "used AdaGrad algorithm" should {
      behave like successfullyTrainedWith(AdaGrad(rate = 0.05f))
    }

    "used Adamax algorithm" should {
      behave like successfullyTrainedWith(Adamax(rate = 0.003f, initAcc = 0.01f))
    }

    "used Nadam algorithm" should {
      behave like successfullyTrainedWith(Nadam())
    }
  }
}
