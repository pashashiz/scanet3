package scanet

import org.apache.spark.sql.Dataset
import scanet.core.Session.withing
import scanet.core.{Expr, Numeric, TF2}
import scanet.math.syntax._
import scanet.models.TrainedModel
import scanet.optimizers.syntax._
import scanet.optimizers.{Record, TensorIterator}

import scala.collection.immutable.Seq

package object estimators {

  def accuracy[A: Numeric](
      model: TrainedModel[A],
      ds: Dataset[Record[A]]): Float = {
    val batchSize = 10000
    val brModel = ds.sparkSession.sparkContext.broadcast(model)
    val shapes = ds.shapes
    val (positives, total) = ds.rdd
      .mapPartitions(it => {
        val model = brModel.value
        val batches = TensorIterator(
          rows = it,
          shapes = shapes,
          batch = batchSize,
          withPadding = false)
        withing(session => {
          val positive = TF2[Expr, A, Expr, A, Expr[Int]]((x, y) => {
            val yPredicted = model.buildResult(x).round
            val matchedOutputs = (y.cast[A] :== yPredicted).cast[Int].sum(Seq(1))
            (matchedOutputs :== model.outputShape(x.shape << 1).last.const).cast[Int].sum
          }).compile(session)
          val result = batches.foldLeft((0, 0))((acc, next) => {
            val (x, y) = next
            val (positiveAcc, totalAcc) = acc
            (positiveAcc + positive(x, y).toScalar, totalAcc + x.shape.head)
          })
          Iterator(result)
        })
      })
      .reduce((left, right) => {
        val (leftPositives, leftSize) = left
        val (rightPositives, rightSize) = right
        (leftPositives + rightPositives, leftSize + rightSize)
      })
    positives.toFloat / total
  }
}
