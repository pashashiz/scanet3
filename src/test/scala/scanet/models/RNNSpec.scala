package scanet.models

import org.scalatest.tags.Slow
import org.scalatest.wordspec.AnyWordSpec
import scanet.core._
import scanet.estimators.{R2Score, RMSE}
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.models.layer.{Dense, LSTMCell, RNN, SimpleRNNCell}
import scanet.optimizers.Adam
import scanet.optimizers.Effect.RecordLoss
import scanet.optimizers.syntax._
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

@Slow
class RNNSpec extends AnyWordSpec with CustomMatchers with SharedSpark with Datasets {

  "RNN neural network" should {

    "train as forecast predictor using Simple RNN Cell" in {
      val display = false
      val Array(train, test) = monthlySunspots(12).randomSplit(Array(0.8, 0.2), 1)
      val model = RNN(SimpleRNNCell(units = 3)) >> Dense(1, Tanh)
      val trained = train
        .train(model)
        .loss(MeanSquaredError)
        .using(Adam())
        .batch(10)
        .each(1.epochs, RecordLoss(tensorboard = true))
        .stopAfter(100.epochs)
        .run()
      val predict = trained.result.compile
      if (display) {
        val expectedBoard = TensorBoard("board/expected")
        val predictedBoard = TensorBoard("board/predicted")
        train.collectTensors().zipWithIndex.foreach {
          case (record, step) =>
            val predicted = predict(record.features)
            expectedBoard.addScalar("forecast", record.labels.slice(0, 0).toScalar, step)
            predictedBoard.addScalar("forecast", predicted.slice(0, 0).toScalar, step)
        }
      }
      RMSE(trained, test) should be < 0.2f
      R2Score(trained, test) should be > 0.8f

      // we might do better abstraction:
      // val predicted = trained.predict(test) - to predict an entire dataset
      // val metrics = trained.metrics(test) - to get access to model metrics
      // metrics.RMSE
      // metrics.R2Score
    }

    "train as forecast predictor using LSTM Cell" in {
      val display = true
      val Array(train, test) = monthlySunspots(1).randomSplit(Array(0.8, 0.2), 1)
      val model = RNN(LSTMCell(units = 3)) >> Dense(1, Tanh)
      val trained = train
        .train(model)
        .loss(MeanSquaredError)
        .using(Adam())
        .batch(10)
        .each(1.epochs, RecordLoss(tensorboard = true))
        .stopAfter(100.epochs)
        .run()
      val predict = trained.result.compile
      println(RMSE(trained, test))
      println(R2Score(trained, test))
      RMSE(trained, test) should be < 0.2f
      R2Score(trained, test) should be > 0.8f
    }
  }
}
