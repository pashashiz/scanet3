package org.scanet

import org.apache.spark.rdd.RDD
import org.scanet.core.Session.withing
import org.scanet.core.{Output, TF2, TensorType}
import org.scanet.math.syntax._
import org.scanet.math.{Floating, Numeric}
import org.scanet.models.TrainedModel
import org.scanet.optimizers.{Optimizer, Tensor2Iterator}

package object estimators {

  def accuracy[A: Floating: Numeric: TensorType](model: TrainedModel[A], ds: RDD[Array[A]]): Float = {
    val batchSize = 10000
    val brModel = ds.sparkContext.broadcast(model)
    val (positives, total) = ds.mapPartitions(it => {
      val model = brModel.value
      val batches = Tensor2Iterator(it, batchSize, splitAt = size => size - model.outputs(), withPadding = false)
      withing(session => {
        val positive = TF2[Output, A, Output, A, Output[Int]](
          (x, y) => {
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
    }).reduce((left, right) => {
      val (leftPositives, leftSize) = left
      val (rightPositives, rightSize) = right
      (leftPositives + rightPositives, leftSize + rightSize)
    })
    positives.toFloat / total
  }
}
