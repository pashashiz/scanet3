package scanet.models.layer

import scanet.core.{Expr, Floating, OutputSeq, Shape}
import scanet.math.nn.ConvFormat._
import scanet.math.nn.Padding._
import scanet.math.nn.{ConvFormat, Padding}
import scanet.math.syntax.zeros
import scanet.syntax._

import scala.collection.immutable.Seq

case class Pool2D(
    window: (Int, Int) = (2, 2),
    strides: (Int, Int) = (1, 1),
    padding: Padding = Valid,
    format: ConvFormat = NHWC)
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
      format = format)
    expr.shape << 1
  }
}
