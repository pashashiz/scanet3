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
  def build[E: Floating](x: Expr[E], weights: Seq[Expr[E]]): Expr[E]

  /** Additional model penalty to be added to the loss
    *
    * @param weights model weights
    * @return penalty
    */
  def penalty[E: Floating](weights: Seq[Expr[E]]): Expr[E]

  def result[E: Floating]: (Expr[E], Seq[Expr[E]]) => Expr[E] =
    (x, w) => build(x, w)

  def outputShape(input: Shape): Shape
  def outputShapeBatched(inputBatched: Shape): Shape = {
    val input = inputBatched << 1
    inputBatched(0) +: outputShape(input)
  }

  def weightsShapes(input: Shape): Seq[Shape]
  def initWeights[E: Floating](input: Shape): Seq[Expr[E]]

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  def displayResult[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    build(
      placeholder[E](inputWithBatch),
      weightsShapes(input).map(s => placeholder[E](s)))
      .as("result")
      .display(dir)
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

  def build[E: Floating](x: Expr[E], y: Expr[E], weights: Seq[Expr[E]]): Expr[E] =
    lossF.build(model.build(x, weights), y) plus model.penalty(weights)

  def loss[E: Floating]: (Expr[E], Expr[E], Seq[Expr[E]]) => Expr[E] =
    (x, y, w) => build(x, y, w)

  def grad[E: Floating]: (Expr[E], Expr[E], Seq[Expr[E]]) => Seq[Expr[E]] =
    (x, y, w) => build(x, y, w).grad(w).returns[E]

  def trained[E: Floating](weights: Seq[Tensor[E]]) = new TrainedModel(this, weights)

  def displayLoss[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    build(
      placeholder[E](inputWithBatch),
      placeholder[E](model.outputShapeBatched(inputWithBatch)),
      model.weightsShapes(input).map(s => placeholder[E](s)))
      .as("loss")
      .display(dir)
  }

  def displayGrad[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    grad[E].apply(
      placeholder[E](inputWithBatch),
      placeholder[E](model.outputShapeBatched(inputWithBatch)),
      model.weightsShapes(input).map(s => placeholder[E](s)))
      .zipWithIndex
      .map {
        case (w, i) => w.as(s"loss_grad_${i}_layer")
      }
      .display(dir)
  }

  override def toString: String = s"$lossF($model)"
}

class TrainedModel[E: Floating](val lossModel: LossModel, val weights: Seq[Tensor[E]]) {

  def buildResult(x: Expr[E]): Expr[E] = lossModel.model.build(x, weights.map(_.const))

  def result: Expr[E] => Expr[E] = (x: Expr[E]) => buildResult(x)

  def buildLoss(x: Expr[E], y: Expr[E]): Expr[E] = lossModel.build(x, y, weights.map(_.const))

  def loss: (Expr[E], Expr[E]) => Expr[E] = (x, y) => buildLoss(x, y)

  def outputShape(input: Shape): Shape = lossModel.model.outputShape(input)
}
