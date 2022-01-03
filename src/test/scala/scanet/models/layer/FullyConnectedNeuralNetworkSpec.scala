package scanet.models.layer

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.{Shape, Tensor, TensorBoard}
import scanet.estimators.accuracy
import scanet.images.Grayscale
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.optimizers.Effect.{RecordAccuracy, RecordLoss}
import scanet.optimizers.syntax._
import scanet.optimizers.{Adam, Record, TRecord}
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class FullyConnectedNeuralNetworkSpec
    extends AnyFlatSpec
    with CustomMatchers
    with SharedSpark
    with Datasets {

  "fully connected neural network with 2 layers (4, 1)" should "minimize logistic regression" in {
    val impl = spark.implicits
    import impl._
    val ds = logisticRegression
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    val trained = ds
      .train(model)
      .loss(BinaryCrossentropy)
      .using(Adam(0.1f))
      .initWith(s => Tensor.zeros(s))
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

  "MNIST dataset" should "be trained with Softmax" ignore {
    val (trainingDs, testDs) = MNIST()
    val model = Dense(50, Sigmoid) >> Dense(10, Softmax)
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
    TensorBoard("board")
      .addImage("layer-1", trained.weights(0).reshape(50, 785, 1), Grayscale())
      .addImage("layer-2", trained.weights(1).reshape(10, 51, 1), Grayscale())
  }
}
