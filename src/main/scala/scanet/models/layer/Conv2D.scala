package scanet.models.layer
import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.nn.ConvFormat._
import scanet.math.nn.Padding._
import scanet.math.nn.{ConvFormat, Padding}
import scanet.math.syntax.zeros
import scanet.models.Activation
import scanet.models.Activation.Identity

import scala.collection.immutable

// todo: RDD -> TensorDataset(featureShape, outputShape, rdd(flattened x + y))
// todo: initializers, regularizers, constraints

// input   = (batch_shape, in_height, in_width, in_channels)          = (1, 5, 5, 1)
// filters = (filter_height, filter_width, in_channels, out_channels) = (2, 2, 1, 1)
// output  = (batch_shape, out_height, out_width, out_channels)       = (1, 5, 5, 1)
case class Conv2D(
    filters: Int,
    kernel: (Int, Int) = (3, 3),
    strides: (Int, Int) = (1, 1),
    padding: Padding = Same,
    format: ConvFormat = NHWC,
    activation: Activation = Identity,
    bias: Boolean = true)
    extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.size == 1, "Conv2D layer can have only one set of weights")
    ???
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())

  // features = in_height x in_width x in_channels
  // todo: well.. here in_height x in_width unknown
  // so the optimizer should know about the input shape...
  override def shapes(features: Int): immutable.Seq[Shape] = ???

  // outputs = out_height x out_width x filter
  override def outputs(): Int = ???

  // todo: better output which represent the shape of the expression without batch dimension
  // input (batch_shape, in_height, in_width, in_channels) -> output (batch_shape, out_height, out_width, filters)
  // outputShape: Shape = (out_height x out_width x filter)
  // outputShapeBatch(batch: Int): Shape = (batch x out_height x out_width x filter)

  // features (in_height, in_width, in_channels) -> weightsShapes Seq((filter_height, filter_width, in_channels, filters))
  // def weightsShapes(features: Shape): Seq[Shape]
}
