package org.scanet.models
import org.scanet.core.{Output, Shape, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.math.syntax._

case class Dense(outputs: Int, activation: Activation) extends Model {

  override def build[E: Numeric : Floating : TensorType]
    (x: Output[E], weights: Output[E]): Output[E] = {
    // x:(n, m) w:(o, m) -> x * w.t -> (n, m) * (m, o) -> (n, o)
    // we have all outputs for each input sample which we will compare with y at the end
    activation.build(withBias(x) * weights.transpose)
  }

  override def shape(features: Int): Shape =
    Shape(outputs, features  + 1)
}

// calc(x, y, weights, meta, iter)
