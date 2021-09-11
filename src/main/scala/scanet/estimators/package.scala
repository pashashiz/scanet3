package scanet

import org.apache.spark.rdd.RDD
import scanet.core.Session.withing
import scanet.core.{Expr, TF2, TensorType}
import scanet.math.syntax._
import scanet.math.{Floating, Numeric}
import scanet.models.TrainedModel
import scanet.optimizers.Tensor2Iterator
import scala.collection.immutable.Seq

package object estimators {

  def accuracy[A: Floating: Numeric: TensorType](
      model: TrainedModel[A],
      ds: RDD[Array[A]]): Float = {
    val batchSize = 10000
    val brModel = ds.sparkContext.broadcast(model)
    val (positives, total) = ds
      .mapPartitions(it => {
        val model = brModel.value
        val batches = Tensor2Iterator(
          it,
          batchSize,
          splitAt = size => size - model.outputs(),
          withPadding = false)
        withing(session => {
          val positive = TF2[Expr, A, Expr, A, Expr[Int]]((x, y) => {
            val yPredicted = model.buildResult(x).round
            val matchedOutputs = (y.cast[A] :== yPredicted).cast[Int].sum(Seq(1))
            (matchedOutputs :== model.outputs().const).cast[Int].sum
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
