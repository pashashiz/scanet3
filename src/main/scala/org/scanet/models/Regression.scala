package org.scanet.models

import org.scanet.core.Slice.syntax.::
import org.scanet.core.{Shape, Tensor}
import org.scanet.math.syntax._

object Regression {

  def linear: Model[Float, Float, Float] = {
    Model[Float, Float, Float](
      (batch, weights) => {
        // batch: (n, m - 1), weights: (m)
        val rows = batch.shape.dims(0)
        val columns = batch.shape.dims(1) - 1
        val x = batch.slice(::, 0 until columns)
        val y = batch.slice(::, columns)
        val bx = Tensor.ones[Float](rows, 1).const.joinAlong(x, 1)
        val wt = weights.reshape(1, columns + 1).transpose
        (0.5f / rows).const :* (bx * wt - y).pow(2).sum
      },
      features => Shape(features))
  }
}
