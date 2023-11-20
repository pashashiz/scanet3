package scanet.math.nn

import scanet.core
import scanet.core.Require.fail
import scanet.core._
import scanet.math.nn.ConvFormat._
import scanet.math.nn.Padding._
import scanet.math.syntax._

import scala.collection.immutable.Seq

sealed trait Padding {

  def shape(format: ConvFormat, input: Shape, reduceWindow: Shape, strides: Seq[Int]): Shape = {
    format.hwAxis.foldLeft(input) { (original, axis) =>
      original.updated(axis, size(format, axis, input(axis), reduceWindow(axis), strides(axis)))
    }
  }

  def size(format: ConvFormat, inputAxis: Int, input: Int, window: Int, stride: Int): Int

  def name: String = getClass.getSimpleName.replace("$", "").toUpperCase
}

object Padding {

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
    override def size(
        format: ConvFormat,
        inputAxis: Int,
        input: Int,
        window: Int,
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
    override def size(
        format: ConvFormat,
        inputAxis: Int,
        input: Int,
        window: Int,
        stride: Int): Int =
      math.ceil((input.toDouble - window + 1) / stride).toInt
  }

  /** Explicit padding
    *
    * @param height `(pad_top, pad_bottom)`
    * @param width `pad_left, pad_right`
    */
  case class Explicit(height: (Int, Int), width: (Int, Int)) extends Padding {

    override def size(
        format: ConvFormat,
        inputAxis: Int,
        input: Int,
        window: Int,
        stride: Int): Int = {
      val padding =
        if (inputAxis == format.hAxis) {
          height._1 + height._2
        } else if (inputAxis == format.wAxis) {
          width._1 + width._2
        } else {
          0
        }
      math.ceil((input.toDouble + padding - window) / stride).toInt + 1
    }

    /** @param format When `format` is
      *  - [[NHWC]] this should be in the form {{{Seq(0, 0, pad_top, pad_bottom, pad_left, pad_right, 0, 0)}}}.
      *  - [[NCHW]] this should be in the form {{{Seq(0, 0, 0, 0, pad_top, pad_bottom, pad_left, pad_right)}}}.
      */
    def flatten(format: ConvFormat): Seq[Int] = {
      val hw = Seq(height._1, height._2, width._1, width._2)
      format match {
        case NHWC            => Seq(0, 0) ++ hw ++ Seq(0, 0)
        case ConvFormat.NCHW => Seq(0, 0, 0, 0) ++ hw
      }
    }
  }
}

sealed trait ConvFormat {
  def name: String = getClass.getSimpleName.replace("$", "")
  def hAxis: Int
  def wAxis: Int
  def hwAxis: Seq[Int] = Seq(hAxis, wAxis)
  def cAxis: Int
  def fill(dense: Seq[Int], others: Int = 1, power: Int = 4, as: String): Seq[Int] = {
    dense match {
      case both +: Nil => Seq.fill(power)(others).updated(hAxis, both).updated(wAxis, both)
      case height +: width +: Nil =>
        Seq.fill(power)(others).updated(hAxis, height).updated(wAxis, width)
      case all if all.size == power => all
      case other                    => fail(s"Either 4, 2 or 1 $as is required, but was passed $other")
    }
  }
  def shapeOf(height: Int, width: Int, others: Int = 1, power: Int = 4): Shape = {
    val base = List.fill(power)(others)
    Shape(base.updated(hAxis, height).updated(wAxis, width))
  }
}

object ConvFormat {

  /** Format in which the input tensor has a form of `batch_shape + [in_height, in_width, in_channels]` */
  case object NHWC extends ConvFormat {
    override def hAxis: Int = 1
    override def wAxis: Int = 2
    override def cAxis: Int = 3
  }

  /** Format in which the input tensor has a form of `batch_shape + [in_channels, in_height, in_width]` */
  case object NCHW extends ConvFormat {
    override def hAxis: Int = 2
    override def wAxis: Int = 3
    override def cAxis: Int = 1
  }
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

  override val shape: Shape = {
    // input   = (batch_shape, in_height, in_width, in_channels)
    // filters = (filter_height, filter_width, in_channels, out_channels)
    // output  = (batch_shape, out_height, out_width, out_channels)
    val convolved = padding.shape(
      format,
      input.shape,
      format.shapeOf(filters.shape(0), filters.shape(1)),
      strides)
    convolved.updated(format.cAxis, filters.shape(3))
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
    new Conv2D(
      input.requireRank(4, as = "input"),
      filters.requireRank(4, as = "filters"),
      format.fill(strides, as = "strides"),
      padding,
      format)
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
  override val inputs: Seq[Expr[_]] = Seq(input, filters.toTensor.const, parentGrad)
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
  override val inputs: Seq[Expr[_]] = Seq(input.toTensor.const, filters, parentGrad)
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

sealed trait Reduce

object Reduce {
  case object Max extends Reduce
  case object Avg extends Reduce
}

case class MaxPool2D[A: Floating] private (
    input: Expr[A],
    window: Shape,
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {
  override def name: String = "MaxPoolV2"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = padding.shape(format, input.shape, window, strides)
  override val inputs: Seq[Expr[_]] =
    Seq(input, window.toTensor.const, Tensor.vector(strides: _*).const)
  override def compiler: core.Compiler[A] = {
    DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("padding", padding.name)
  }
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val inputGrad =
        MaxPool2DGradInput[R](
          input.cast[R],
          current.cast[R],
          parentGrad,
          window,
          strides,
          padding,
          format)
      Seq(inputGrad)
    }
  }
}

case class MaxPool2DGradInput[A: Floating](
    input: Expr[A],
    output: Expr[A],
    parentGrad: Expr[A],
    window: Shape,
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {
  override def name: String = "MaxPoolGradV2"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = input.shape
  override val inputs: Seq[Expr[_]] =
    Seq(input, output, parentGrad, window.toTensor.const, Tensor.vector(strides: _*).const)
  override def compiler: core.Compiler[A] = {
    DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("padding", padding.name)
  }
}

case class AvgPool2D[A: Floating] private (
    input: Expr[A],
    window: Shape,
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {
  override def name: String = "AvgPool"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = padding.shape(format, input.shape, window, strides)
  override val inputs: Seq[Expr[_]] = Seq(input)
  override def compiler: core.Compiler[A] = {
    DefaultCompiler[A]()
      .withAttr("ksize", window.toLongArray)
      .withAttr("strides", strides.map(_.toLong))
      .withAttr("data_format", format.name)
      .withAttr("padding", padding.name)
  }
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val inputGrad =
        AvgPool2DGradInput[R](input.shape, parentGrad, window, strides, padding, format)
      Seq(inputGrad)
    }
  }
}

case class AvgPool2DGradInput[A: Floating](
    input: Shape,
    parentGrad: Expr[A],
    window: Shape,
    strides: Seq[Int],
    padding: Padding,
    format: ConvFormat)
    extends Expr[A] {
  override def name: String = "AvgPoolGrad"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = input
  override val inputs: Seq[Expr[_]] = Seq(input.toTensor.const, parentGrad)
  override def compiler: core.Compiler[A] = {
    DefaultCompiler[A]()
      .withAttr("ksize", window.toLongArray)
      .withAttr("strides", strides.map(_.toLong))
      .withAttr("data_format", format.name)
      .withAttr("padding", padding.name)
  }
}

object Pool2D {

  def apply[A: Floating](
      input: Expr[A],
      window: Seq[Int] = Seq(2),
      strides: Seq[Int] = Seq(1),
      padding: Padding = Same,
      format: ConvFormat = NHWC,
      reduce: Reduce = Reduce.Max): Expr[A] = {
    val supportedPadding: Set[Padding] = Set(Same, Valid)
    require(
      supportedPadding.contains(padding),
      s"only [${supportedPadding.mkString(", ")}] paddings are supported")
    val inputChecked = input.requireRank(4, as = "input")
    val windowFilled = Shape(format.fill(window, as = "window"): _*)
    val stridesFilled = format.fill(strides, as = "strides")
    reduce match {
      case Reduce.Max => MaxPool2D(inputChecked, windowFilled, stridesFilled, padding, format)
      case Reduce.Avg => AvgPool2D(inputChecked, windowFilled, stridesFilled, padding, format)
    }
  }
}

case class FusedBatchNorm[A: Floating](
    input: Expr[A],
    scale: Expr[A],
    offset: Expr[A],
    mean: Expr[A],
    variance: Expr[A],
    format: ConvFormat,
    training: Boolean,
    epsilon: Option[Float] = None,
    exponentialAvgFactor: Option[Float] = None)
    extends Expr[A] {
  override def name: String = "FusedBatchNormV3"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = input.shape
  override val inputs: Seq[Expr[_]] = Seq(input, scale, offset, mean, variance)
  override def compiler: core.Compiler[A] = {
    // find out good wat to update state with seq of optional values
    val comp1 = DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("is_training", training)
    val comp2 = epsilon.fold(comp1)(e => comp1.withAttr("epsilon", e))
    val comp3 = exponentialAvgFactor.fold(comp2)(e => comp1.withAttr("exponential_avg_factor", e))
    comp3
  }
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R: Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val outs = outputs
      val grads = FusedBatchNormGrad[R](
        input.cast[R],
        parentGrad,
        scale.cast[R],
        outs.reserveSpace1.cast[R],
        outs.reserveSpace2.cast[R],
        outs.reserveSpace3.cast[R],
        format,
        training,
        epsilon).outputs
      Seq(grads.inputGrad, grads.scaleGrad, grads.offsetGrad)
    }
  }
  def outputs: FusedBatchNormOutputs[A] = {
    // reserveSpace1 + reserveSpace2 + reserveSpace3 shape?
    val channelShape = mean.shape
    FusedBatchNormOutputs(
      output = this,
      batchMean = TakeOut[A](this, 1, channelShape),
      batchVariance = TakeOut[A](this, 1, channelShape),
      reserveSpace1 = TakeOut[A](this, 2, channelShape),
      reserveSpace2 = TakeOut[A](this, 3, channelShape),
      reserveSpace3 = TakeOut[A](this, 4, channelShape))
  }
}

trait BatchNormOutputs[A] {
  def output: Expr[A]
  def batchMean: Expr[A]
  def batchVariance: Expr[A]
  def mapMean(f: Expr[A] => Expr[A]): BatchNormOutputs[A]
  def mapVariance(f: Expr[A] => Expr[A]): BatchNormOutputs[A]
}

/** Outputs of [[FusedBatchNorm]] operator
  * @param output A 4D Output Tensor
  * @param batchMean A 1D Tensor for the computed batch mean, to be used by TensorFlow to compute the running mean.
  * @param batchVariance A 1D Tensor for the computed batch variance, to be used by TensorFlow to compute the running variance.
  * @param reserveSpace1 A 1D Tensor for the computed batch mean, to be reused in the gradient computation.
  * @param reserveSpace2 A 1D Tensor for the computed batch variance (inverted variance in the cuDNN case), to be reused in the gradient computation.
  * @param reserveSpace3 A 1D Tensor for some intermediate results, to be reused in the gradient computation for better efficiency.
  */
case class FusedBatchNormOutputs[A](
    output: Expr[A],
    batchMean: Expr[A],
    batchVariance: Expr[A],
    reserveSpace1: Expr[A],
    reserveSpace2: Expr[A],
    reserveSpace3: Expr[A])
    extends BatchNormOutputs[A] {
  override def mapMean(f: Expr[A] => Expr[A]): BatchNormOutputs[A] =
    copy(batchMean = f(batchMean))
  override def mapVariance(f: Expr[A] => Expr[A]): BatchNormOutputs[A] =
    copy(batchVariance = f(batchVariance))
}

/** Outputs of batch norm
  * @param output An Output Tensor
  * @param batchMean A Tensor for the computed batch mean, to be used by TensorFlow to compute the running mean.
  * @param batchVariance A Tensor for the computed batch variance, to be used by TensorFlow to compute the running variance.
  */
case class StdBatchNormOutputs[A](
    output: Expr[A],
    batchMean: Expr[A],
    batchVariance: Expr[A])
    extends BatchNormOutputs[A] {
  override def mapMean(f: Expr[A] => Expr[A]): BatchNormOutputs[A] =
    copy(batchMean = f(batchMean))
  override def mapVariance(f: Expr[A] => Expr[A]): BatchNormOutputs[A] =
    copy(batchVariance = f(batchVariance))
}

/** @param input A 4D Tensor for input data
  * @param parentGrad A 4D Tensor for the gradient with respect to output
  * @param scale A 1D Tensor for scaling factor, to scale the normalized input.
  * @param reserveSpace1 When `training` is `true`, a 1D Tensor for the computed batch mean to be reused in gradient computation.
  *                      When `training` is `false`, a 1D Tensor for the population mean to be reused
  *                      in both 1st and 2nd order gradient computation.
  * @param reserveSpace2 When `training` is `true`, a 1D Tensor for the computed batch variance
  *                      (inverted variance in the cuDNN case) to be reused in gradient computation.
  *                      When `training` is `false`, a 1D Tensor for the population variance to be reused
  *                      in both 1st and 2nd order gradient computation.
  * @param reserveSpace3 When `training` is `true`, a 1D Tensor for some intermediate results to be reused in gradient computation.
  *                      When `training` is `false`, a dummy empty Tensor will be created.
  * @param format  One of [[NCHW]] or [[NHWC]]
  * @param training A bool value to indicate the operation is for training (default) or inference.
  * @param epsilon A small float number added to the variance of input
  */
case class FusedBatchNormGrad[A: Floating](
    input: Expr[A],
    parentGrad: Expr[A],
    scale: Expr[A],
    reserveSpace1: Expr[A],
    reserveSpace2: Expr[A],
    reserveSpace3: Expr[A],
    format: ConvFormat,
    training: Boolean,
    epsilon: Option[Float] = None)
    extends Expr[A] {
  override def name: String = "FusedBatchNormGradV3"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = input.shape
  override val inputs: Seq[Expr[_]] =
    Seq(parentGrad, input, scale, reserveSpace1, reserveSpace2, reserveSpace3)
  override def compiler: core.Compiler[A] = {
    val comp1 = DefaultCompiler[A]()
      .withAttr("data_format", format.name)
      .withAttr("is_training", training)
    epsilon.fold(comp1)(e => comp1.withAttr("epsilon", e))
  }
  def outputs: FusedBatchNormGradOutputs[A] = {
    val channelShape = scale.shape
    FusedBatchNormGradOutputs(
      inputGrad = this,
      scaleGrad = TakeOut[A](this, 1, channelShape),
      offsetGrad = TakeOut[A](this, 2, channelShape))
  }
}

/** Outputs of [[FusedBatchNormGrad]] operator
  * @param inputGrad A 4D Tensor for the gradient with respect to input
  * @param scaleGrad  A 1D Tensor for the gradient with respect to scale
  * @param offsetGrad A 1D Tensor for the gradient with respect to offset
  */
case class FusedBatchNormGradOutputs[A](inputGrad: Expr[A], scaleGrad: Expr[A], offsetGrad: Expr[A])

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
      format: ConvFormat = NHWC): Expr[A] =
    Conv2D(input, filters, strides, padding, format)

  /** Performs the pooling (reduction of sliding window) on the input tensor with a given function [[Reduce.Max]] or [[Reduce.Avg]]
    *
    * @param input A 4-D Tensor of the format specified by `format`.
    * @param window The size of the window for each dimension of the input tensor to perform pooling,
    *               should be a sequence of strides that has length 1, 2 or 4.
    *               If a single value is given it is replicated in the H and W dimension.
    *               By default the N and C dimensions are set to 1.
    * @param strides The stride of the sliding window for each dimension of input,
    *                should be a sequence of strides that has length 1, 2 or 4.
    *                If a single value is given it is replicated in the H and W dimension.
    *                By default the N and C dimensions are set to 1. The dimension order is determined
    *                by the value of `format`, see below for details.
    * @param padding Padding algorithm to use or explicit paddings at the start and end of each dimension.
    *                See [[https://www.tensorflow.org/api_docs/python/tf/nn#notes_on_padding_2) here]] for more information.
    * @param format Specifies whether the channel dimension of the input and output is the last dimension, see [[NHWC]] and [[NCHW]]
    * @param reduce Reduction function either [[Reduce.Max]] or [[Reduce.Avg]]
    * @return The pooled output tensor. The shape depends on `format` and `padding` (same as for [[conv2D]]
    */
  def pool2D[A: Floating](
      input: Expr[A],
      window: Seq[Int] = Seq(1),
      strides: Seq[Int] = Seq(1),
      padding: Padding = Same,
      format: ConvFormat = NHWC,
      reduce: Reduce = Reduce.Max): Expr[A] =
    Pool2D(input, window, strides, padding, format, reduce)

  /** Batch normalization. Note that the size of 4D Tensors are defined by either [[NCHW]] or [[NHWC]].
    *
    * @param input        A 4D Tensor for input data
    * @param scale        A 1D Tensor for scaling factor to scale the normalized input
    * @param offset       A 1D Tensor for offset, to shift to the normalized input
    * @param mean         A 1D Tensor for population mean. Used for inference only; must be empty for training
    * @param variance     A 1D Tensor for population variance. Used for inference only; must be empty for training
    * @param format       One of [[NCHW]] or [[NHWC]]
    * @param training     A bool value to indicate the operation is for training (default) or inference.
    * @param epsilon      A small float number added to the variance of input
    * @param expAvgFactor The exponential avg factor
    * @return Outputs
    */
  def fusedBatchNorm[A: Floating](
      input: Expr[A],
      scale: Expr[A],
      offset: Expr[A],
      mean: Expr[A],
      variance: Expr[A],
      format: ConvFormat,
      training: Boolean,
      epsilon: Option[Float] = None,
      expAvgFactor: Option[Float] = None): FusedBatchNormOutputs[A] = {
    // format: off
    FusedBatchNorm(input, scale, offset, mean, variance, format, training, epsilon, expAvgFactor).outputs
    // format: on
  }

  /** Batch normalization. Supports input of any shape
    *
    * @param input Tensor for input data
    * @param scale Tensor for scaling factor to scale the normalized input
    * @param offset Tensor for offset, to shift to the normalized input
    * @param mean A Tensor for population mean. Used for inference only; must be empty for training
    * @param variance A Tensor for population variance. Used for inference only; must be empty for training
    * @param training A bool value to indicate the operation is for training (default) or inference.
    * @param axes Axes that should be normalized (typically the features axes).
    * @param epsilon A small float number added to the variance of input
    * @return output
    */
  def batchNorm[A: Floating](
      input: Expr[A],
      scale: Expr[A],
      offset: Expr[A],
      mean: Expr[A],
      variance: Expr[A],
      training: Boolean,
      axes: Seq[Int] = Seq(-1),
      epsilon: Float = 1e-3f): StdBatchNormOutputs[A] = {
    val (batchMean, batchVariance) =
      if (training) {
        val reduceAxis = input.shape.axesExcept(axes: _*)
        input.moments(reduceAxis, keepDims = true)
      } else {
        (mean, variance)
      }
    val epsilonE = epsilon.const.cast[A]
    // use rsqrt instead of sqrt to increase performance ~20%
    val inv = rsqrt(batchVariance + epsilonE) * scale
    val output = ((input * inv) - (batchMean * inv)) + offset
    StdBatchNormOutputs(output, batchMean, batchVariance)
  }
}

object kernels {
  trait AllSyntax extends AllKernels {}
  object syntax extends AllSyntax
}
