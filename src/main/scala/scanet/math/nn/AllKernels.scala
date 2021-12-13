package scanet.math.nn

import scanet.core
import scanet.core.syntax._
import scanet.core.Require.fail
import scanet.core.{DefaultCompiler, Expr, Floating, Grad, Shape, Tensor, TensorType}

import scala.collection.immutable.Seq

sealed trait Padding {
  def shape(format: ConvFormat, inputAxis: Int, input: Int, filter: Int, stride: Int): Int
  def name: String = getClass.getSimpleName.replace("$", "").toUpperCase
}

/** Padding is applied to each spatial dimension.
  * When the strides are 1, the input is padded such that the output size is the same as the input size.
  * In the 2D case, the output size is computed as:
  *
  * {{{
  * out_height = ceil(in_height / stride_height)
  * out_width  = ceil(in_width / stride_width)
  * }}}
  *
  * The amount of padding used is the smallest amount that results in the output size.
  * Note that top/bottom and right/left paddings might be different
  */
case object Same extends Padding {
  override def shape(
      format: ConvFormat,
      inputAxis: Int,
      input: Int,
      filter: Int,
      stride: Int): Int =
    math.ceil(input.toDouble / stride).toInt
}

/** No padding to be used. This causes the output size to typically be smaller than the input size,
  * even when the stride is one. In the 2D case, the output size is computed as:
  *
  * {{{
  * out_height = ceil((in_height - filter_height + 1) / stride_height)
  * out_width  = ceil((in_width - filter_width + 1) / stride_width)
  * }}}
  */
case object Valid extends Padding {
  override def shape(
      format: ConvFormat,
      inputAxis: Int,
      input: Int,
      filter: Int,
      stride: Int): Int =
    math.ceil((input.toDouble - filter + 1) / stride).toInt
}

/** Explicit padding
  *
  * @param height `(pad_top, pad_bottom)`
  * @param width `pad_left, pad_right`
  */
case class Explicit(height: (Int, Int), width: (Int, Int)) extends Padding {

  override def shape(
      format: ConvFormat,
      inputAxis: Int,
      input: Int,
      filter: Int,
      stride: Int): Int = {
    val padding =
      if (inputAxis == format.hAxis) {
        height._1 + height._2
      } else if (inputAxis == format.wAxis) {
        width._1 + width._2
      } else {
        0
      }
    math.ceil((input.toDouble + padding - filter) / stride).toInt + 1
  }

  /** @param format When `format` is
    *  - [[NHWC]] this should be in the form {{{Seq(0, 0, pad_top, pad_bottom, pad_left, pad_right, 0, 0)}}}.
    *  - [[NCHW]] this should be in the form {{{Seq(0, 0, 0, 0, pad_top, pad_bottom, pad_left, pad_right)}}}.
    */
  def flatten(format: ConvFormat): Seq[Int] = {
    val hw = Seq(height._1, height._2, width._1, width._2)
    format match {
      case NHWC => Seq(0, 0) ++ hw ++ Seq(0, 0)
      case NCHW => Seq(0, 0, 0, 0) ++ hw
    }
  }
}

sealed trait ConvFormat {
  def name: String = getClass.getSimpleName.replace("$", "")
  def hAxis: Int
  def wAxis: Int
  def hwAxis: Seq[Int] = Seq(hAxis, wAxis)
}

/** Format in which the input tensor has a form of `batch_shape + [in_height, in_width, in_channels]` */
case object NHWC extends ConvFormat {
  override def hAxis: Int = 1
  override def wAxis: Int = 2
}

/** Format in which the input tensor has a form of `batch_shape + [in_channels, in_height, in_width]` */
case object NCHW extends ConvFormat {
  override def hAxis: Int = 2
  override def wAxis: Int = 3
}

case class Conv2D[A: Floating] private (
    input: Expr[A],
    filters: Expr[A],
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {

  override def name: String = "Conv2D"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])

  override val shape: Shape =
    format.hwAxis.zip(Seq(0, 1)).foldLeft(input.shape) { (shape, axis) =>
      val (inputAxis, filterAxis) = axis
      shape.updated(
        inputAxis,
        padding.shape(
          format,
          inputAxis,
          input.shape.get(inputAxis),
          filters.shape(filterAxis),
          strides(inputAxis)))
    }

  override def inputs: Seq[Expr[_]] = Seq(input, filters)

  override def compiler: core.Compiler[A] = {
    val compiler = DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("strides", strides.map(_.toLong))
      .withAttr("padding", padding.name)
      .withAttr("use_cudnn_on_gpu", value = true)
    padding match {
      case explicit: Explicit =>
        compiler.withAttr("explicit_paddings", explicit.flatten(format).map(_.toLong))
      case _ => compiler
    }
  }

  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val inputGrad =
        Conv2DGradInput(input.shape, filters.cast[R], parentGrad, strides, padding, format)
      val filterGrad =
        Conv2DGradFilter(input.cast[R], filters.shape, parentGrad, strides, padding, format)
      Seq(inputGrad, filterGrad)
    }
  }
}

object Conv2D {

  def apply[A: Floating](
      input: Expr[A],
      filters: Expr[A],
      strides: Seq[Int] = Seq(1),
      padding: Padding = Same,
      format: ConvFormat = NHWC): Expr[A] = {
    new Conv2D(input, filters, normalizeStrides(strides, format), padding, format)
  }

  def normalizeStrides(strides: Seq[Int], format: ConvFormat): Seq[Int] =
    strides match {
      case both +: Nil => Seq.fill(4)(1).updated(format.hAxis, both).updated(format.wAxis, both)
      case height +: width +: Nil =>
        Seq.fill(4)(1).updated(format.hAxis, height).updated(format.wAxis, width)
      case all if all.size == 4 => all
      case other                => fail(s"Either 4, 2 or 1 strides is required, but was passed $other")
    }
}

case class Conv2DGradFilter[A: Floating](
    input: Expr[A],
    filters: Shape,
    parentGrad: Expr[A],
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {
  override def name: String = "Conv2DBackpropFilter"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = filters
  override def inputs: Seq[Expr[_]] =
    Seq(input, Tensor.vector(filters.dims.toArray).const, parentGrad)
  override def compiler: core.Compiler[A] = {
    val compiler = DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("strides", strides.map(_.toLong))
      .withAttr("padding", padding.name)
      .withAttr("use_cudnn_on_gpu", value = true)
    padding match {
      case explicit: Explicit =>
        compiler.withAttr("explicit_paddings", explicit.flatten(format).map(_.toLong))
      case _ => compiler
    }
  }
}

case class Conv2DGradInput[A: Floating](
    input: Shape,
    filters: Expr[A],
    parentGrad: Expr[A],
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {
  override def name: String = "Conv2DBackpropInput"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = input
  override def inputs: Seq[Expr[_]] =
    Seq(Tensor.vector(input.dims.toArray).const, filters, parentGrad)
  override def compiler: core.Compiler[A] = {
    val compiler = DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("strides", strides.map(_.toLong))
      .withAttr("padding", padding.name)
      .withAttr("use_cudnn_on_gpu", value = true)
    padding match {
      case explicit: Explicit =>
        compiler.withAttr("explicit_paddings", explicit.flatten(format).map(_.toLong))
      case _ => compiler
    }
  }
}

trait AllKernels {

  /** Computes a 2-D convolution given input and 4-D filters tensors.
    *
    * The input tensor may have rank 4 or higher,
    * where shape dimensions `[:-3]` are considered batch dimensions (batch shape).
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
    * @param input A Tensor of rank at least 4. The dimension order is interpreted according to the value of `format`;
    *              with the all-but-inner-3 dimensions acting as batch dimensions. See below for details.
    * @param filters A 4-D tensor of shape `[filter_height, filter_width, in_channels, out_channels]`
    * @param strides The stride of the sliding window for each dimension of input,
    *                should be a sequence of strides that has length 1, 2 or 4.
    *                If a single value is given it is replicated in the H and W dimension.
    *                By default the N and C dimensions are set to 1. The dimension order is determined
    *                by the value of `format`, see below for details.
    * @param padding Padding algorithm to use or explicit paddings at the start and end of each dimension.
    *                See [[https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2) here]] for more information.
    * @param format Specifies whether the channel dimension of the input and output is the last dimension, see [[NHWC]] and [[NCHW]]
    * @return A Tensor of shape
    *  - when format is [[NHWC]]: `[batch_size] + output_spatial_shape + [out_channels]`
    *  - when format is [[NCHW]]: `[batch_size, out_channels] + output_spatial_shape`
    *  - when padding [[Same]]: `output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides[i])`
    *  - when padding [[Valid]]: `output_spatial_shape[i] = ceil((input_spatial_shape[i] - (spatial_filter_shape[i] + 1)) / strides[i])`
    */
  def conv2D[A: Floating](
      input: Expr[A],
      filters: Expr[A],
      strides: Seq[Int] = Seq(1),
      padding: Padding = Same,
      format: ConvFormat = NHWC): Expr[A] = Conv2D(input, filters, strides, padding, format)
}
