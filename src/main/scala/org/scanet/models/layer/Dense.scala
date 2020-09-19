package org.scanet.models.layer

import org.scanet.core.{Output, OutputSeq, Shape, TensorType}
import org.scanet.math.{Floating, Numeric}
import org.scanet.models.Activation
import org.scanet.syntax._

/**
 * Regular densely-connected NN layer.
 *
 * Dense implements the operation:
 * {{{output = activation(input * weights.t) + bias)}}}
 * where:
 * - `activation` is the element-wise activation function passed as the activation argument
 * - `weights` is a weights matrix created by the layer
 * - `bias` is a bias vector created by the layer
 *
 * @param outputs neurons of a dense layer, if used as a last layer
 *                it will be equal to number of possible classification groups
 * @param activation function to apply to each output (neuron)
 */
case class Dense(outputs: Int, activation: Activation, bias: Float = 1f) extends Layer {

  override def build[E: Numeric : Floating : TensorType](x: Output[E], weights: OutputSeq[E]) = {
    require(weights.size == 1, "Dense layer can have only one set of weights")
    // x:(samples, features)
    // w:(outputs, features)
    // x * w.t -> (samples, features) * (features, outputs) -> (samples, outputs)
    activation.build(withBias(x, bias.const.cast[E]) matmul weights.head.transpose)
  }

  override def shapes(features: Int) = Seq(Shape(outputs, features  + 1))
}
