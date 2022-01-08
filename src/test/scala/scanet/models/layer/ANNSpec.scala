package scanet.models.layer

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.{Shape, Tensor}
import scanet.estimators.accuracy
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.optimizers.{Adam, TRecord}
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class ANNSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "fully connected neural network with 2 layers (4, 1)" should "minimize logistic regression" in {
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
    model.displayResult[Float](input = Shape(3))
  }

  it should "produce right graph of a loss function given x shape" ignore {
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    model.withLoss(BinaryCrossentropy).displayLoss[Float](input = Shape(3))
  }

  it should "produce right graph of loss gradient given x shape" ignore {
    val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
    model.withLoss(BinaryCrossentropy).displayGrad[Float](Shape(3))
  }
}
