package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.models.Initializer.{GlorotUniform, Zeros}
import scanet.models.Regularization.Zero
import scanet.models.{Activation, Initializer, Regularization}
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
    * @param kernelInitializer kernel initializer
    * @param biasInitializer bias initializer
    */
  def apply(
      outputs: Int,
      activation: Activation,
      reg: Regularization = Zero,
      bias: Boolean = true,
      kernelInitializer: Initializer = GlorotUniform(),
      biasInitializer: Initializer = Zeros): Layer = {
    val dense = new Dense(outputs, reg, kernelInitializer)
    dense ?>> (bias, Bias(outputs, reg, biasInitializer)) ?>> (activation.ni, activation.layer)
  }
}

case class Dense private (outputs: Int, reg: Regularization, initializer: Initializer)
    extends StatelessLayer {

  override def build[E: Floating](input: Expr[E], weights: Seq[Expr[E]]) = {
    require(weights.size == 1, "Dense layer can have only one set of weights")
    // x:(samples, features)
    // w:(features, outputs)
    // x * w -> (samples, features) * (features, outputs) -> (samples, outputs)
    input matmul weights.head
  }

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]) =
    reg.build(weights.head)

  override def weightsShapes(input: Shape): Seq[Shape] = {
    require(input.rank == 2, "features should have a shape (batch, features)")
    Seq(Shape(input(1), outputs))
  }

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] =
    Seq(initializer.build[E](weightsShapes(input).head))

  override def outputShape(input: Shape): Shape = Shape(input.head, outputs)

}
