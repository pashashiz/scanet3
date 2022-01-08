package scanet.models

import scanet.core._
import scanet.math.syntax._

import scala.collection.immutable.Seq

abstract class Model extends Serializable {

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

  def weightsCount: Int = 1
  def weightsShapes(input: Shape): Seq[Shape]
  def trainable: Boolean = weightsCount > 0

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  def withBias[E: Floating](x: Expr[E], bias: Expr[E]): Expr[E] = {
    val rows = x.shape.dims.head
    fillOutput[E](rows, 1)(bias).joinAlong(x, 1)
  }

  def displayResult[E: Floating](input: Shape, dir: String = ""): Unit = {
    val inputWithBatch = input >>> 1
    result[E].display(
      Seq(inputWithBatch),
      weightsShapes(input),
      label = "result",
      dir = dir)
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

  override def toString: String = s"$model:$lossF"
}

class TrainedModel[E: Floating](val lossModel: LossModel, val weights: Seq[Tensor[E]]) {

  def buildResult(x: Expr[E]): Expr[E] = lossModel.model.build(x, weights.map(_.const))

  def result: TF1[E, Tensor[E], Expr[E]] = TF1(buildResult)

  def buildLoss(x: Expr[E], y: Expr[E]): Expr[E] = lossModel.build(x, y, weights.map(_.const))

  def loss: TF2[E, Tensor[E], E, Tensor[E], Expr[E]] =
    TF2[Expr, E, Expr, E, Expr[E]]((x, y) => buildLoss(x, y))

  def outputShape(input: Shape): Shape = lossModel.model.outputShape(input)
}
