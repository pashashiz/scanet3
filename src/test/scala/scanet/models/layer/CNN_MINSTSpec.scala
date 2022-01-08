package scanet.models.layer

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.tags.Slow
import scanet.core.{Numeric, Tensor, TensorBoard}
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
class CNN_MINSTSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "MNIST dataset" should "be trained with CNN" in {
    val (trainingDs, testDs) = MNIST()
    val model =
      Conv2D(32, activation = ReLU()) >> Pool2D() >>
      Conv2D(64, activation = ReLU()) >> Pool2D() >>
      Flatten() >> Dense(10, Softmax)
    val trained = trainingDs
      .train(model)
      .loss(CategoricalCrossentropy)
      .using(Adam(0.001f))
      .batch(100)
      .initWith(shape => Tensor.rand(shape, range = Some(-0.1f, 0.1f)))
      .each(1.epochs, RecordLoss())
      .each(1.epochs, RecordAccuracy(testDs))
      .stopAfter(3.epochs)
      .run()
    println(accuracy(trained, testDs))
    accuracy(trained, testDs) should be >= 0.98f
  }
}
