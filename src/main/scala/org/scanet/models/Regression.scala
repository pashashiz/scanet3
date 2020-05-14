package org.scanet.models

import org.scanet.math.syntax._
import org.scanet.core.Slice.syntax.::

class Regression {

  def linear(): Model[Float, Float, Float] = {
    Model[Float, Float, Float](batch =>
      TensorFunction(weights => {
        // weights = (k)
        // batch = (n, k + 1)
        // batch.slice(::, 0 until )
        // todo: need slice and concat
        ???
      }))
  }
}
