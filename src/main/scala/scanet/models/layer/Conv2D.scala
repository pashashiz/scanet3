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

// todo:
//  - initializers
//  - regularizers
//  - constraints
//  - dilation_rate

// input   = (batch_shape, in_height, in_width, in_channels)          = (1, 5, 5, 1)
// filters = (filter_height, filter_width, in_channels, out_channels) = (2, 2, 1, 1)
// output  = (batch_shape, out_height, out_width, out_channels)       = (1, 5, 5, 1)
case class Conv2D(
    filters: Int,
    kernel: (Int, Int) = (3, 3),
    strides: (Int, Int) = (1, 1),
    padding: Padding = Valid,
    format: ConvFormat = NHWC,
    activation: Activation = Identity,
    bias: Boolean = true)
    extends Layer {

  def filterHeight: Int = kernel._1
  def filterWidth: Int = kernel._2

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.size == 1, "Conv2D layer can have only one set of weights")
    val convolved = conv2D[E](
      input = x,
      filters = weights.head,
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format)
    activation.build(convolved)
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
