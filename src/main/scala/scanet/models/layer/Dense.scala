package scanet.models.layer

import scanet.core.Params.Weights
import scanet.core.{Expr, Floating, Params, Shape}
import scanet.models.Aggregation.Avg
import scanet.models.Initializer.{GlorotUniform, Zeros}
import scanet.models.Regularization.Zero
import scanet.models.{Activation, Initializer, ParamDef, Regularization}
import scanet.syntax._

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
      activation: Activation = Activation.Identity,
      reg: Regularization = Zero,
      bias: Boolean = true,
      kernelInitializer: Initializer = GlorotUniform(),
      biasInitializer: Initializer = Zeros,
      trainable: Boolean = true): Layer = {
    val dense = new Dense(outputs, reg, kernelInitializer, trainable)
    dense ?>> (bias, Bias(outputs, reg, biasInitializer)) ?>> (activation.ni, activation.layer)
  }
}

case class Dense private (
    outputs: Int,
    reg: Regularization,
    initializer: Initializer,
    override val trainable: Boolean)
    extends StatelessLayer {

  override def params(input: Shape): Params[ParamDef] =
    Params(Weights -> ParamDef(
      Shape(input(1), outputs),
      initializer,
      Some(Avg),
      trainable = trainable))

  override def buildStateless[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] =
    // x:(samples, features)
    // w:(features, outputs)
    // x * w -> (samples, features) * (features, outputs) -> (samples, outputs)
    input matmul params.weights

  override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] =
    reg.build(params.weights)

  override def outputShape(input: Shape): Shape = Shape(input.head, outputs)

  override def makeTrainable(trainable: Boolean): Dense = copy(trainable = trainable)

  override def toString: String = s"Dense($outputs)"
}
