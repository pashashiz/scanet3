package scanet.models.layer

import scanet.core.Params.Weights
import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.nn.ConvFormat._
import scanet.math.nn.Padding._
import scanet.math.nn.{ConvFormat, Padding}
import scanet.math.syntax.zeros
import scanet.models.{Activation, Initializer, ParamDef, Regularization}
import scanet.models.Activation.Identity
import scanet.models.Aggregation.Avg
import scanet.models.Initializer.{GlorotUniform, Zeros}
import scanet.models.Regularization.Zero
import scanet.syntax._

import scala.collection.immutable.Seq

object Conv2D {

  /** This layer creates a 2-D convolution kernel that is convolved with the layer input
    * to produce a tensor of outputs (see [[conv2D]] kernel).
    *
    * The input tensor may have rank 4 or higher,
    * where shape dimensions `(:-3)` are considered batch dimensions (batch shape).
    * The dimension order is interpreted according to the value of [[format]];
    * with the all-but-inner-3 dimensions acting as batch dimensions. See below for details.
    *
    * Given an input tensor of shape `batch_shape + [in_height, in_width, in_channels]` and
    * a filter / kernel tensor of shape `[filter_height, filter_width, in_channels, out_channels]`,
    * this op performs the following:
    *
    *  - Flattens the filter to a 2-D matrix with shape
    *    `[filter_height * filter_width * in_channels, output_channels]`.
    *  - Extracts image patches from the input tensor to form a virtual tensor of shape
    *    `[batch, out_height, out_width, filter_height * filter_width * in_channels]`.
    *  - For each patch, right-multiplies the filter matrix and the image patch vector.
    *
    * If `bias = true`, a bias vector is created and added to the outputs.
    * Finally, if activation is not `Identity`, it is applied to the outputs as well.
    *
    * The resulting output shape depends on `format` and `padding` (see [[conv2D]]
    *
    * @param filters the dimensionality of the output space (i.e. the number of output filters in the convolution).
    * @param kernel `(height, width)` of the 2D convolution window (filter)
    * @param strides `(height, width)` specifying the strides of the convolution along the height and width
    * @param padding Padding algorithm to use or explicit paddings at the start and end of each dimension.
    *                See [[https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2) here]] for more information.
    * @param format Specifies whether the channel dimension of the input and output is the last dimension, see [[NHWC]] and [[NCHW]]
    * @param activation Activation function to use
    * @param bias Whether to add bias vector
    * @param kernelInitializer kernel initializer
    * @param biasInitializer bias initializer
    */
  def apply(
      filters: Int,
      kernel: (Int, Int) = (3, 3),
      strides: (Int, Int) = (1, 1),
      padding: Padding = Valid,
      format: ConvFormat = NHWC,
      activation: Activation = Identity,
      bias: Boolean = false,
      kernelInitializer: Initializer = GlorotUniform(),
      biasInitializer: Initializer = Zeros,
      biasReg: Regularization = Zero,
      trainable: Boolean = true): Layer = {
    val conv = new Conv2D(filters, kernel, strides, padding, format, kernelInitializer, trainable)
    conv ?>> (bias, Bias(filters, biasReg, biasInitializer)) ?>> (activation.ni, activation.layer)
  }
}

// todo: regularizers, dilation_rate, constraints
case class Conv2D private (
    filters: Int,
    kernel: (Int, Int),
    strides: (Int, Int),
    padding: Padding,
    format: ConvFormat,
    initializer: Initializer,
    override val trainable: Boolean)
    extends StatelessLayer {

  def filterHeight: Int = kernel._1
  def filterWidth: Int = kernel._2

  override def params(input: Shape): Params[ParamDef] = {
    require(
      input.rank == 4,
      s"Conv2D input should have a shape (NHWC) or (NCHW) but was $input")
    val shape = Shape(filterHeight, filterWidth, input(format.cAxis), filters)
    Params(Weights -> ParamDef(shape, initializer, Some(Avg), trainable = trainable))
  }

  override def buildStateless[E: Floating](input: Expr[E], params: Params[Expr[E]]): Expr[E] = {
    // Conv2D example:
    // input   = (batch_shape, in_height, in_width, in_channels)          = (1, 5, 5, 1)
    // filters = (filter_height, filter_width, in_channels, out_channels) = (2, 2, 1, 1)
    // output  = (batch_shape, out_height, out_width, out_channels)       = (1, 5, 5, 1)
    conv2D[E](
      input = input,
      filters = params.weights,
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format)
  }

  override def penalty[E: Floating](params: Params[Expr[E]]): Expr[E] =
    zeros[E](Shape())

  override def outputShape(input: Shape): Shape = {
    require(
      input.rank == 4,
      s"Conv2D input should have a shape (NHWC) or (NCHW) but was $input")
    val expr = conv2D[Float](
      input = zeros[Float](input),
      filters = zeros[Float](filterHeight, filterWidth, input(format.cAxis), filters),
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format)
    expr.shape
  }

  override def makeTrainable(trainable: Boolean): Conv2D = copy(trainable = trainable)

  override def toString: String = s"Conv2D($filters,$kernel,$strides,$padding,$format)"
}
