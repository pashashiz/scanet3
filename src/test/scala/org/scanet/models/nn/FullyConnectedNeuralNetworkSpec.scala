package org.scanet.models.nn

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.{Shape, Tensor, TensorBoard}
import org.scanet.datasets.MNIST
import org.scanet.estimators.accuracy
import org.scanet.images.Grayscale
import org.scanet.models.{BinaryCrossentropy, Sigmoid}
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.optimizers.{Adam, Tensor2Iterator}
import org.scanet.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class FullyConnectedNeuralNetworkSpec extends AnyFlatSpec with CustomMatchers  with SharedSpark with Datasets {

  "fully connected neural network with 2 layers (4, 1)" should "minimize logistic regression" in {
    val ds = logisticRegression.map(a => Array(a(0)/100, a(1)/100, a(2)))
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    val trained = ds.train(model)
      .loss(BinaryCrossentropy)
      .using(Adam(0.1f))
      .initWith(s => Tensor.zeros(s))
      .batch(100)
      .each(1.epochs, logResult())
      .stopAfter(50.epochs)
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

  it should "produce right graph of a result function given x shape" ignore {
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    model.displayResult[Float](x = Shape(4, 3))
  }

  it should "produce right graph of a loss function given x shape" ignore {
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    model.withLoss(BinaryCrossentropy).displayLoss[Float](x = Shape(4, 3))
  }

  it should "produce right graph of loss gradient given x shape" ignore {
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    model.withLoss(BinaryCrossentropy).displayGrad[Float](x = Shape(4, 3))
  }

  "MNIST dataset" should "be trained" ignore {
    val (trainingDs, testDs) = MNIST.load(sc)
    val model = Dense(50, Sigmoid) >> Dense(10, Sigmoid)
    val trained = trainingDs.train(model)
      .loss(BinaryCrossentropy)
      .using(Adam(0.01f))
      .initWith(s => Tensor.zeros(s))
      .batch(1000)
      .each(1.epochs, logResult())
      .stopAfter(300.epochs)
      .run()
    accuracy(trained, testDs) should be >= 0.87f
  }
}
