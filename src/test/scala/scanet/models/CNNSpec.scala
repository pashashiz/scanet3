package scanet.models

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Shape
import scanet.estimators.accuracy
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.models.layer.{Activate, Conv2D, Dense, Flatten, Pool2D}
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
        Conv2D(32, activation = ReLU()) >> Pool2D(strides = (2, 2)) >>
        Conv2D(64, activation = ReLU()) >> Pool2D(strides = (2, 2)) >>
        Activate(ReLU()) >> Flatten >> Dense(64, ReLU()) >> Dense(10, Softmax)
      val trained = trainingDs
        .train(model)
        .loss(CategoricalCrossentropy)
        .using(Adam())
        .batch(1000)
        .each(1.epochs, RecordLoss())
        .each(1.epochs, RecordAccuracy(testDs))
        .stopAfter(10.epochs)
        .run()
      accuracy(trained, testDs) should be >= 0.98f
    }

    "be described" in {
      val model =
        Conv2D(32, activation = ReLU()) >> Pool2D() >>
        Conv2D(64, activation = ReLU()) >> Pool2D() >>
        Flatten >> Dense(10, Softmax)
      val expected =
        """#+-----------------------------------------------+--------------+------+------------+
           #|name                                           |weights       |params|output      |
           #+-----------------------------------------------+--------------+------+------------+
           #|Input                                          |              |      |(_,24,24,1) |
           #|Conv2D(32,(3,3),(1,1),Valid,NHWC,GlorotUniform)|(3, 3, 1, 32) |288   |(_,22,22,32)|
           #|ReLU(0.0)                                      |              |      |(_,22,22,32)|
           #|Pool2D((2,2),(1,1),Valid,NHWC,Max)             |              |      |(_,21,21,32)|
           #|Conv2D(64,(3,3),(1,1),Valid,NHWC,GlorotUniform)|(3, 3, 32, 64)|18432 |(_,19,19,64)|
           #|ReLU(0.0)                                      |              |      |(_,19,19,64)|
           #|Pool2D((2,2),(1,1),Valid,NHWC,Max)             |              |      |(_,18,18,64)|
           #|Flatten                                        |              |      |(_,20736)   |
           #|Dense(10,Zero,GlorotUniform)                   |(20736, 10)   |207360|(_,10)      |
           #|Bias(10,Zero,Zeros)                            |(10)          |10    |(_,10)      |
           #|Softmax                                        |              |      |(_,10)      |
           #+-----------------------------------------------+--------------+------+------------+
           #Total params: 226090 (883.2 KB)"""
          .stripMargin('#')
      println(model.describe[Float](Shape(1, 24, 24, 1)))
      model.describe[Float](Shape(1, 24, 24, 1)) shouldBe expected
    }
  }
}
