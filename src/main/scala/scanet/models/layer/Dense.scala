package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.models.Regularization.Zero
import scanet.models.{Activation, Regularization}
import scanet.syntax._

import scala.collection.immutable.Seq

object Dense {

  /** Regular densely-connected NN layer.
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
    * @param reg regularization
    * @param bias whether to add bias
    */
  def apply(
      outputs: Int,
      activation: Activation,
      reg: Regularization = Zero,
      bias: Boolean = true): Layer = {
    val dense = new Dense(outputs, reg)
    dense ?>> (bias, Bias(outputs, reg)) ?>> (activation.ni, activation.layer)
  }
}

case class Dense private (outputs: Int, reg: Regularization) extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]) = {
    require(weights.size == 1, "Dense layer can have only one set of weights")
    // x:(samples, features)
    // w:(outputs, features)
    // x * w.t -> (samples, features) * (features, outputs) -> (samples, outputs)
    x matmul weights.head.transpose
  }

  override def penalty[E: Floating](weights: OutputSeq[E]) =
    reg.build(weights.head)

  override def weightsShapes(input: Shape): Seq[Shape] = {
    require(input.rank == 1, "features should have a shape (features)")
    Seq(Shape(outputs, input(0)))
  }

  override def outputShape(input: Shape): Shape = Shape(outputs)
}
