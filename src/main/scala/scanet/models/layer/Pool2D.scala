package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.nn.ConvFormat._
import scanet.math.nn.Padding._
import scanet.math.nn.{ConvFormat, Padding, Reduce}
import scanet.math.syntax.zeros
import scanet.syntax._

import scala.collection.immutable.Seq

/** Pooling operation for 2D spatial data.
  *
  * Downsamples the input along its spatial dimensions (`height` and `width`)
  * by taking the [[Reduce.Max]] or [[Reduce.Avg]] value over an input window
  * (of size defined by pool_size) for each channel of the input.
  * The window is shifted by strides along each dimension.
  *
  * The resulting output shape depends on `format` and `padding` (see [[pool2D]]
  *
  * @param window `(height, width)` of the window to apply pooling
  * @param strides `(height, width)` specifies how far the pooling window moves for each pooling step
  * @param padding Padding algorithm to use or explicit paddings at the start and end of each dimension.
  *                See [[https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2) here]] for more information.
  * @param format  Specifies whether the channel dimension of the input and output is the last dimension, see [[NHWC]] and [[NCHW]]
  * @param reduce  Reduction function either [[Reduce.Max]] or [[Reduce.Avg]]
  */
case class Pool2D(
    window: (Int, Int) = (2, 2),
    strides: (Int, Int) = (1, 1),
    padding: Padding = Valid,
    format: ConvFormat = NHWC,
    reduce: Reduce = Reduce.Max)
    extends Layer {

  override def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E] = {
    require(weights.isEmpty, "Pool2D layer does not require weights")
    pool2D[E](
      input = x,
      window = Seq(window._1, window._2),
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format)
  }

  override def penalty[E: Floating](weights: OutputSeq[E]): Expr[E] = zeros[E](Shape())

  override def weightsCount: Int = 0

  override def weightsShapes(input: Shape): Seq[Shape] = Seq.empty

  override def outputShape(input: Shape): Shape = {
    require(
      input.rank == 3,
      s"Pool2D input should have a shape (in_height, in_width, in_channels) but was $input")
    val expr = pool2D[Float](
      // we add batch dimension to a head
      // (in_height, in_width, in_channels) >>> 1 -> (1, in_height, in_width, in_channels)
      input = zeros[Float](input >>> 1),
      // since there is no batch in input we << 1
      window = Seq(window._1, window._2),
      strides = Seq(strides._1, strides._2),
      padding = padding,
      format = format,
      reduce = reduce)
    expr.shape << 1
  }
}
