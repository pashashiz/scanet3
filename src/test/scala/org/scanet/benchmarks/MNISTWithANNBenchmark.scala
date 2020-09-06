package org.scanet.benchmarks

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import org.scanet.estimators.accuracy
import org.scanet.math.syntax._
import org.scanet.models.Activation.{Sigmoid, Softmax}
import org.scanet.models.Loss.CategoricalCrossentropy
import org.scanet.models.layer.Dense
import org.scanet.optimizers.Effect.{RecordAccuracy, RecordLoss}
import org.scanet.optimizers._
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

trait MNISTWithANNBehaviours {

  this: AnyWordSpec with CustomMatchers with SharedSpark with Datasets =>

  def successfullyTrainedWith(alg: Algorithm): Unit = {
    "be successfully trained" in {
      val (trainingDs, testDs) = MNIST()
      val model = Dense(50, Sigmoid) >> Dense(10, Softmax)
      val trained = trainingDs.train(model)
        .loss(CategoricalCrossentropy)
        .using(alg)
        .batch(1000)
        .each(1.epochs, RecordLoss(tensorboard = true, dir = s"board/$alg"))
        .each(5.epochs, RecordAccuracy(testDs, tensorboard = true, dir = s"board/$alg"))
        .stopAfter(50.epochs)
        .run()
      accuracy(trained, testDs) should be >= 0.9f
    }
  }
}

@Slow
class MNISTWithANNBenchmark extends AnyWordSpec with CustomMatchers with SharedSpark with Datasets with MNISTWithANNBehaviours {

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
