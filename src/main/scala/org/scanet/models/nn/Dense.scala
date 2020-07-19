package org.scanet.models.nn

import org.scanet.core.{Output, OutputSeq, Shape, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.models.Activation
import org.scanet.syntax._

case class Dense(outputs: Int, activation: Activation) extends Layer {

  override def build[E: Numeric : Floating : TensorType](x: Output[E], weights: OutputSeq[E]) = {
    require(weights.size == 1, "Dense layer can have only one set of weights")
    // x:(samples, features)
    // w:(outputs, features)
    // x * w.t -> (samples, features) * (features, outputs) -> (samples, outputs)
    activation.build(withBias(x) * weights.head.transpose)
  }

  override def shapes(features: Int) = Seq(Shape(outputs, features  + 1))
}
