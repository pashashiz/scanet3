package scanet.models.layer

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.tags.Slow
import scanet.core.TensorBoard
import scanet.estimators.accuracy
import scanet.images.Grayscale
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.optimizers.Adam
import scanet.optimizers.Effect.{RecordAccuracy, RecordLoss}
import scanet.optimizers.syntax._
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

@Slow
class ANN_MINSTSpec
    extends AnyFlatSpec
    with CustomMatchers
    with SharedSpark
    with Datasets {

  "MNIST dataset" should "be trained with ANN" in {
    val (trainingDs, testDs) = MNIST()
    val model = Flatten() >> Dense(50, Sigmoid) >> Dense(10, Softmax)
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
