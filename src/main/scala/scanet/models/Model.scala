package scanet.models

import scanet.core._
import scanet.math.syntax._
import scanet.utils.{Bytes, Tabulator}

import scala.collection.immutable.Seq

abstract class Model extends Serializable {

  def name: String = getClass.getSimpleName.replace("$", "")

  /** Build a model
    *
    * @param x training set, where first dimension equals to number of samples (batch size)
    * @param weights model weights
    * @return model
    */
  def build[E: Floating](x: Expr[E], weights: OutputSeq[E]): Expr[E]

  /** Additional model penalty to be added to the loss
    *
    * @param weights model weights
    * @return penalty
    */
  def penalty[E: Floating](weights: OutputSeq[E]): Expr[E]

  def result[E: Floating]: TF2[E, Tensor[E], E, Seq[Tensor[E]], Expr[E]] =
    TF2[Expr, E, OutputSeq, E, Expr[E]](build[E])

  def outputShape(input: Shape): Shape
  def outputShapeBatched(inputBatched: Shape): Shape = {
    val input = inputBatched << 1
    inputBatched(0) +: outputShape(input)
  }

  def weightsShapes(input: Shape): Seq[Shape]
  def initWeights[E: Floating](input: Shape): OutputSeq[E]

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  def displayResult[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    result[E].display(
      Seq(inputWithBatch),
      weightsShapes(input),
      label = "result",
      dir = dir)
  }

  def info(input: Shape): Seq[LayerInfo] =
    Seq(LayerInfo(name, weightsShapes(input).headOption, outputShape(input)))

  def describe[E: Floating](input: Shape): String = {
    val layersInfo = info(input)
    val layers = (LayerInfo("Input", None, input) +: layersInfo).map(_.toRow)
    val table = Tabulator.format(Seq("name", "weights", "params", "output") +: layers)
    val total = layersInfo.map(info => info.weights.map(_.power).getOrElse(0)).sum
    val size = Bytes.formatSize(TensorType[E].codec.sizeOf(total))
    s"$table\nTotal params: $total ($size)"
  }
}

case class LayerInfo(name: String, weights: Option[Shape], output: Shape) {
  def toRow: Seq[String] = {
    val weightsStr = weights.map(_.toString).getOrElse("")
    val params = weights.map(_.power.toString).getOrElse("")
    val outputStr = ("_" +: output.dims.map(_.toString)).mkString("(", ",", ")")
    Seq(name, weightsStr, params, outputStr)
  }
}

case class LossModel(model: Model, lossF: Loss) extends Serializable {

  def build[E: Floating](x: Expr[E], y: Expr[E], weights: OutputSeq[E]): Expr[E] =
    lossF.build(model.build(x, weights), y) plus model.penalty(weights)

  def loss[E: Floating]: TF3[E, Tensor[E], E, Tensor[E], E, Seq[Tensor[E]], Expr[E]] =
    TF3[Expr, E, Expr, E, OutputSeq, E, Expr[E]](build[E])

  def weightsAndGrad[E: Floating]
      : TF3[E, Tensor[E], E, Tensor[E], E, Seq[Tensor[E]], (OutputSeq[E], OutputSeq[E])] =
    TF3[Expr, E, Expr, E, OutputSeq, E, (OutputSeq[E], OutputSeq[E])]((x, y, w) =>
      (w, build(x, y, w).grad(w).returns[E]))

  def grad[E: Floating]: TF3[E, Tensor[E], E, Tensor[E], E, Seq[Tensor[E]], OutputSeq[E]] =
    TF3[Expr, E, Expr, E, OutputSeq, E, OutputSeq[E]]((x, y, w) =>
      build(x, y, w).grad(w).returns[E])

  def trained[E: Floating](weights: Seq[Tensor[E]]) = new TrainedModel(this, weights)

  def displayLoss[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    loss[E].display(
      Seq(inputWithBatch),
      Seq(model.outputShapeBatched(inputWithBatch)),
      model.weightsShapes(input),
      label = "loss",
      dir = dir)
  }

  def displayGrad[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    grad[E].display(
      Seq(inputWithBatch),
      Seq(model.outputShapeBatched(inputWithBatch)),
      model.weightsShapes(input),
      label = "loss_grad",
      dir = dir)
  }

  override def toString: String = s"$lossF($model)"
}

class TrainedModel[E: Floating](val lossModel: LossModel, val weights: Seq[Tensor[E]]) {

  def buildResult(x: Expr[E]): Expr[E] = lossModel.model.build(x, weights.map(_.const))

  def result: TF1[E, Tensor[E], Expr[E]] = TF1(buildResult)

  def buildLoss(x: Expr[E], y: Expr[E]): Expr[E] = lossModel.build(x, y, weights.map(_.const))

  def loss: TF2[E, Tensor[E], E, Tensor[E], Expr[E]] =
    TF2[Expr, E, Expr, E, Expr[E]]((x, y) => buildLoss(x, y))

  def outputShape(input: Shape): Shape = lossModel.model.outputShape(input)
}
