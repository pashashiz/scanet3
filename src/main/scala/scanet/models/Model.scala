package scanet.models

import scanet.core._
import scanet.math.Floating
import scanet.math.Numeric
import scanet.math.syntax._
import scala.collection.immutable.Seq

abstract class Model extends Serializable {

  /**
   * Build a model
   *
   * @param x training set, where first dimension equals to number of amples (batchh size)
   * @param weights model weights
   * @return model
   */
  def build[E: Numeric: Floating: TensorType](x: Expr[E], weights: OutputSeq[E]): Expr[E]

  /**
   * Additional model penalty to be added to the loss
   *
   * @param weights model weights
   * @return penalty
   */
  def penalty[E: Numeric: Floating: TensorType](weights: OutputSeq[E]) : Expr[E]

  def result[E: Numeric: Floating: TensorType]: TF2[E, Tensor[E], E, Seq[Tensor[E]], Expr[E]] =
    TF2[Expr, E, OutputSeq, E, Expr[E]](build[E])

  /** @param features number of features in a dataset
   * @return shape of weights tensor for each layer
   */
  def shapes(features: Int): Seq[Shape]

  /** @return number of model outputs
   */
  def outputs(): Int

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  def withBias[E: Numeric: Floating: TensorType](x: Expr[E], bias: Expr[E]): Expr[E] = {
    val rows = x.shape.dims.head
    fillOutput[E](rows, 1)(bias).joinAlong(x, 1)
  }

  def inferShapeOfY(x: Shape) = {
    val samples = x.dims.head
    Shape(samples, outputs())
  }

  def inferShapeOfWeights(x: Shape) = {
    val features = x.dims(1)
    shapes(features)
  }

  def displayResult[E: Numeric: Floating: TensorType](x: Shape, dir: String = ""): Unit = {
    result[E].display(Seq(x), inferShapeOfWeights(x), label = "result", dir = dir)
  }
}

case class LossModel(model: Model, lossF: Loss) extends Serializable {

  def build[E: Numeric: Floating: TensorType](x: Expr[E], y: Expr[E], weights: OutputSeq[E]): Expr[E] =
    lossF.build(model.build(x, weights), y) plus model.penalty(weights)

  def loss[E: Numeric: Floating: TensorType]: TF3[E, Tensor[E], E, Tensor[E], E, Seq[Tensor[E]], Expr[E]] =
    TF3[Expr, E, Expr, E, OutputSeq, E, Expr[E]](build[E])

  def weightsAndGrad[E: Numeric: Floating: TensorType]: TF3[E, Tensor[E], E, Tensor[E], E, Seq[Tensor[E]], (OutputSeq[E], OutputSeq[E])] =
    TF3[Expr, E, Expr, E, OutputSeq, E, (OutputSeq[E], OutputSeq[E])](
      (x, y, w) => (w, build(x, y, w).grad(w).returns[E]))

  def grad[E: Numeric: Floating: TensorType]: TF3[E, Tensor[E], E, Tensor[E], E, Seq[Tensor[E]], OutputSeq[E]] =
    TF3[Expr, E, Expr, E, OutputSeq, E, OutputSeq[E]](
      (x, y, w) => build(x, y, w).grad(w).returns[E])

  def trained[E: Numeric: Floating: TensorType](weights: Seq[Tensor[E]]) = new TrainedModel(this, weights)

  def displayLoss[E: Numeric: Floating: TensorType](x: Shape, dir: String = ""): Unit = {
    loss[E].display(Seq(x), Seq(model.inferShapeOfY(x)), model.inferShapeOfWeights(x), label = "loss", dir = dir)
  }

  def displayGrad[E: Numeric: Floating: TensorType](x: Shape, dir: String = ""): Unit = {
    grad[E].display(Seq(x), Seq(model.inferShapeOfY(x)), model.inferShapeOfWeights(x), label = "loss_grad", dir = dir)
  }

  override def toString: String = s"$model:$lossF"
}

class TrainedModel[E: Numeric : Floating: TensorType](val lossModel: LossModel, val weights: Seq[Tensor[E]]) {

  def buildResult(x: Expr[E]): Expr[E] = lossModel.model.build(x, weights.map(_.const))

  def result: TF1[E, Tensor[E], Expr[E]] = TF1(buildResult)

  def buildLoss(x: Expr[E], y: Expr[E]): Expr[E] = lossModel.build(x, y, weights.map(_.const))

  def loss: TF2[E, Tensor[E], E, Tensor[E], Expr[E]] =
    TF2[Expr, E, Expr, E, Expr[E]]((x, y) => buildLoss(x, y))

  def outputs(): Int = lossModel.model.outputs()
}