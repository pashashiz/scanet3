package scanet.models

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import scanet.core._
import scanet.estimators.accuracy
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.models.layer.{Dense, Flatten}
import scanet.optimizers.Effect.{RecordAccuracy, RecordLoss}
import scanet.optimizers.syntax._
import scanet.optimizers.{Adam, TRecord}
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

@Slow
class ANNSpec extends AnyWordSpec with CustomMatchers with SharedSpark with Datasets {

  "fully connected neural network" should {

    "minimize logistic regression" in {
      val ds = logisticRegression
      val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
      val trained = ds
        .train(model)
        .loss(BinaryCrossentropy)
        .using(Adam(0.2f))
        .batch(100)
        .each(1.epochs, RecordLoss())
        .stopAfter(50.epochs)
        .run()
      val TRecord(x, y) = ds.firstTensor(100)
      val loss = trained.loss.compile()
      loss(x, y).toScalar should be <= 0.4f
      accuracy(trained, ds) should be >= 0.9f
      val predictor = trained.result.compile()
      val input = Tensor.matrix(Array(0.3462f, 0.7802f), Array(0.6018f, 0.8630f))
      predictor(input).const.round.eval should be(Tensor.matrix(Array(0f), Array(1f)))
    }

    "train on MNIST dataset" in {
      val (trainingDs, testDs) = MNIST()
      val model = Flatten >> Dense(50, Sigmoid) >> Dense(10, Softmax)
      val trained = trainingDs
        .train(model)
        .loss(CategoricalCrossentropy)
        .using(Adam(0.01f))
        .batch(1000)
        .each(1.epochs, RecordLoss())
        .each(10.epochs, RecordAccuracy(testDs))
        .stopAfter(25.epochs)
        .run()
      accuracy(trained, testDs) should be >= 0.95f

//      import scanet.images.Grayscale
//      TensorBoard("board")
//        .addImage("1#Dense", trained.weights(0).reshape(50, 784, 1), Grayscale())
//        .addImage("2#Bias", trained.weights(1).reshape(1, 50, 1), Grayscale())
//        .addImage("3#Dense", trained.weights(2).reshape(10, 50, 1), Grayscale())
//        .addImage("4#Bias", trained.weights(3).reshape(1, 10, 1), Grayscale())
    }

    "produce right graph of a result function given x shape" ignore {
      val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
      model.displayResult[Float](input = Shape(3), "board")
    }

    "produce right graph of a loss function given x shape" ignore {
      val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
      model.withLoss(BinaryCrossentropy).displayLoss[Float](input = Shape(3), "board")
    }

    "produce right graph of loss gradient given x shape" ignore {
      val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
      model.withLoss(BinaryCrossentropy).displayGrad[Float](Shape(3), "board")
    }
  }
}
