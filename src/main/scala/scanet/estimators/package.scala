package scanet

import org.apache.spark.sql.Dataset
import scanet.core.{Expr, Numeric, Session, Shape}

import scala.{math => m}
import scanet.math.syntax._
import scanet.models.TrainedModel
import scanet.optimizers.Iterators.Partial
import scanet.optimizers.syntax._
import scanet.optimizers.Record

import scala.collection.immutable.Seq

// todo: caching
package object estimators {

  def accuracy[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]],
      batch: Int = 1000): Float = {
    import ds.sparkSession.implicits._
    val brModel = ds.sparkSession.sparkContext.broadcast(model)
    val (positives, total) = ds
      .mapPartitionsTensors(batch, remaining = Partial) { it =>
        val model = brModel.value
        val session = new Session()
        val positive = (x: Expr[A], y: Expr[A]) => {
          val yPredicted = model.buildResult(x).round
          val matchedOutputs = (y.cast[A] :== yPredicted).cast[Int].sum(Seq(1))
          (matchedOutputs :== model.outputShape(x.shape).last.const).cast[Int].sum
        }
        val positiveCompiled = positive.compileWith(session)
        val result = it.using(session).foldLeft((0, 0))((acc, next) => {
          val (x, y) = next
          val (positiveAcc, totalAcc) = acc
          (positiveAcc + positiveCompiled(x, y).toScalar, totalAcc + x.shape.head)
        })
        Iterator(result)
      }
      .reduce((left, right) => {
        val (leftPositives, leftSize) = left
        val (rightPositives, rightSize) = right
        (leftPositives + rightPositives, leftSize + rightSize)
      })
    positives.toFloat / total
  }

  def RMSE[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]],
      batch: Int = 1000): Float =
    m.sqrt(MSE(model, ds, batch)).toFloat

  def MSE[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]],
      batch: Int = 1000): Float =
    meanError(model, ds, batch) {
      (predicted, expected) => (predicted - expected).sqr
    }

  def MAE[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]],
      batch: Int = 1000): Float =
    meanError(model, ds, batch) {
      (predicted, expected) => (predicted - expected).abs
    }

  private def meanError[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]],
      batch: Int)(
      error: (Expr[A], Expr[A]) => Expr[A]): Float = {
    import ds.sparkSession.implicits._
    val brModel = ds.sparkSession.sparkContext.broadcast(model)
    val (sum, total) = ds
      .mapPartitionsTensors(batch, remaining = Partial) { it =>
        val model = brModel.value
        val session = new Session()
        val errorSum = (input: Expr[A], expected: Expr[A]) => {
          val predicted = model.buildResult(input)
          error(predicted, expected).sum.cast[Float]
        }
        val errorSumCompiled = errorSum.compileWith(session)
        val result = it.using(session).foldLeft((0f, 0)) {
          case ((sum, size), (input, expected)) =>
            val nextSum = errorSumCompiled(input, expected)
            (sum + nextSum.toScalar, size + input.shape.head)
        }
        Iterator(result)
      }
      .reduce { (left, right) =>
        val (leftSum, leftSize) = left
        val (rightSUm, rightSize) = right
        (leftSum + rightSUm, leftSize + rightSize)
      }
    sum / total
  }

  def R2Score[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]],
      batch: Int = 1000): Float = {
    require(ds.labelsShape == Shape(1), "labels should have shape (1)")
    import ds.sparkSession.implicits._
    val brModel = ds.sparkSession.sparkContext.broadcast(model)
    val expectedVsPredicted = ds
      .mapPartitionsTensors(batch, remaining = Partial) { it =>
        val model = brModel.value
        val session = new Session()
        val predict = model.result.compileWith(session)
        it.using(session).flatMap {
          case (input, expected) =>
            val predicted = predict(input)
            (0 until input.shape.head).map { i =>
              (
                expected.slice(i, 0).toScalar.cast[Float],
                predicted.slice(i, 0).toScalar.cast[Float])
            }
        }
      }
      .cache()
    val mean = expectedVsPredicted.map(_._2).rdd.mean()
    val (ssr, sst) = expectedVsPredicted
      .map {
        case (expected, predicted) =>
          (m.pow(predicted - expected, 2), m.pow(predicted - mean, 2))
      }
      .reduce { (left, right) =>
        val (ssr1, sst1) = left
        val (ssr2, sst2) = right
        (ssr1 + ssr2, sst1 + sst2)
      }
    (1 - (ssr / sst)).toFloat
  }
}
