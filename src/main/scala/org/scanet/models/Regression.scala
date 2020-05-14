package org.scanet.models

import org.scanet.math.syntax._

class Regression {

  def linear(): Model[Float, Float, Float] = {
    Model[Float, Float, Float](x =>
      TensorFunction(w => {
        // todo: need slice...
        ???
      }))
  }
}
