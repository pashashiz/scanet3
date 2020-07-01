package org.scanet

import org.apache.spark.rdd.RDD
import org.scanet.core.Session.withing
import org.scanet.core.{Output, TF2, Tensor}
import org.scanet.math.syntax._
import org.scanet.models.TrainedModel
import org.scanet.optimizers.Tensor2Iterator

package object estimators {

  def accuracy(model: TrainedModel[Float, Float, Float], ds: RDD[Array[Float]]): Float = {
    val batchSize = 10000
    val brModel = ds.sparkContext.broadcast(model)
    val (positives, total) = ds.mapPartitions(it => {
      val model = brModel.value
      val batches = Tensor2Iterator(it, batchSize, splitAt = size => size - model.outputs(), withPadding = false)
      withing(session => {
        val positive = TF2((x: Output[Float], y: Output[Float]) => {
          val yPredicted = model.buildResult(x).round
          (y :== yPredicted).cast[Int].sum
        }).returns[Tensor[Int]].compile(session)
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
