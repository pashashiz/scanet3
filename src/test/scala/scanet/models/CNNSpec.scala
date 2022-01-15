package scanet.models

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import scanet.estimators.accuracy
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.models.layer.{Conv2D, Dense, Flatten, Pool2D}
import scanet.optimizers.Adam
import scanet.optimizers.Effect.{RecordAccuracy, RecordLoss}
import scanet.optimizers.syntax._
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

@Slow
class CNNSpec extends AnyWordSpec with CustomMatchers with SharedSpark with Datasets {

  "convolutional neural network" should {
    "train on MNIST dataset" in {
      val (trainingDs, testDs) = MNIST()
      val model =
        Conv2D(32, activation = ReLU()) >> Pool2D() >>
        Conv2D(64, activation = ReLU()) >> Pool2D() >>
        Flatten >> Dense(10, Softmax)
      val trained = trainingDs
        .train(model)
        .loss(CategoricalCrossentropy)
        .using(Adam())
        .batch(100)
        .each(1.epochs, RecordLoss())
        .each(1.epochs, RecordAccuracy(testDs))
        .stopAfter(3.epochs)
        .run()
      accuracy(trained, testDs) should be >= 0.985f
    }
  }
}
