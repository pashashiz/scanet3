package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.nn.ConvFormat._
import scanet.math.nn.Padding._
import scanet.math.nn.{ConvFormat, Padding}
import scanet.math.syntax.zeros
import scanet.models.Activation
import scanet.models.Activation.Identity
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
    */
  def apply(
      filters: Int,
      kernel: (Int, Int) = (3, 3),
      strides: (Int, Int) = (1, 1),
      padding: Padding = Valid,
      format: ConvFormat = NHWC,
      activation: Activation = Identity,
      bias: Boolean = false): Layer = {
    val conv = new Conv2D(filters, kernel, strides, padding, format)
    conv ?>> (bias, Bias(filters)) ?>> (activation.ni, activation.layer)
  }
}

// todo: initializers, regularizers, constraints, dilation_rate
case class Conv2D private (
    filters: Int,
    kernel: (Int, Int),
    strides: (Int, Int),
    padding: Padding,
    format: ConvFormat)
    extends Layer {

  def filterHeight: Int = kernel._1
  def filterWidth: Int = kernel._2

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.size == 1, "Conv2D layer can have only one set of weights")
    // Conv2D example:
    // input   = (batch_shape, in_height, in_width, in_channels)          = (1, 5, 5, 1)
    // filters = (filter_height, filter_width, in_channels, out_channels) = (2, 2, 1, 1)
    // output  = (batch_shape, out_height, out_width, out_channels)       = (1, 5, 5, 1)
    conv2D[E](
      input = x,
      filters = weights.head,
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format)
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())

  override def weightsShapes(input: Shape): Seq[Shape] = {
    require(
      input.rank == 3,
      s"Conv2D features should have a shape (in_height, in_width, in_channels) but was $input")
    Seq(Shape(filterHeight, filterWidth, input(format.cAxis - 1), filters))
  }

  override def outputShape(input: Shape): Shape = {
    require(
      input.rank == 3,
      s"Conv2D input should have a shape (in_height, in_width, in_channels) but was $input")
    val expr = conv2D[Float](
      // we add batch dimension to a head
      // (in_height, in_width, in_channels) >>> 1 -> (1, in_height, in_width, in_channels)
      input = zeros[Float](input >>> 1),
      // since there is no batch in input we << 1
      filters = zeros[Float](filterHeight, filterWidth, input(format.cAxis - 1), filters),
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format)
    expr.shape << 1
  }
}