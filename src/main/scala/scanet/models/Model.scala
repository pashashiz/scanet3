package scanet.models

import scanet.core._
import scanet.math.syntax._
import scanet.utils.{Bytes, Tabulator}

import scala.collection.immutable.Seq

abstract class Model extends Serializable {

  def name: String = getClass.getSimpleName.replace("$", "")

  def build[E: Floating](input: Expr[E], weights: Seq[Expr[E]]): Expr[E] =
    buildStateful(input, weights, stateShapes(input.shape).map(s => zeros[E](s)))._1

  /** Build a model
    *
    * @param input   training set, where first dimension equals to number of samples (batch size)
    * @param weights model weights
    * @param state model state
    * @return model
    */
  def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]])

  /** Additional model penalty to be added to the loss
    *
    * @param weights model weights
    * @return penalty
    */
  def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E]

  def result[E: Floating]: (Expr[E], Seq[Expr[E]]) => Expr[E] =
    (input, weights) =>
      buildStateful(input, weights, stateShapes(input.shape).map(s => zeros[E](s)))._1

  def resultStateful[E: Floating]
      : (Expr[E], Seq[Expr[E]], Seq[Expr[E]]) => (Expr[E], Seq[Expr[E]]) =
    (input, weights, state) => buildStateful(input, weights, state)

  def outputShape(input: Shape): Shape

  def weightsShapes(input: Shape): Seq[Shape]
  def initWeights[E: Floating](input: Shape): Seq[Expr[E]]

  def stateShapes(input: Shape): Seq[Shape]

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  def displayResult[E: Floating](input: Shape, dir: String = ""): Unit = {
    build(
      placeholder[E](input),
      weightsShapes(input).map(s => placeholder[E](s)))
      .as("result")
      .display(dir)
  }

  def info(input: Shape): Seq[LayerInfo] =
    Seq(LayerInfo(toString, weightsShapes(input), outputShape(input)))

  def describe[E: Floating](input: Shape): String = {
    val layersInfo = info(input)
    val layers = (LayerInfo("Input", Seq.empty, input) +: layersInfo).map(_.toRow)
    val table = Tabulator.format(Seq("name", "weights", "params", "output") +: layers)
    val total = layersInfo.flatMap(info => info.weights.map(_.power)).sum
    val size = Bytes.formatSize(TensorType[E].codec.sizeOf(total))
    s"$table\nTotal params: $total ($size)"
  }
}

case class LayerInfo(name: String, weights: Seq[Shape], output: Shape) {
  def toRow: Seq[String] = {
    val weightsStr = weights.map(_.toString).mkString(", ")
    val params = weights.map(_.power.toString).mkString(", ")
    val outputStr = ("_" +: output.tail.dims.map(_.toString)).mkString("(", ",", ")")
    Seq(name, weightsStr, params, outputStr)
  }
}

case class LossModel(model: Model, lossF: Loss) extends Serializable {

  def build[E: Floating](input: Expr[E], output: Expr[E], weights: Seq[Expr[E]]): Expr[E] =
    lossF.build(model.build(input, weights), output) plus model.penalty(input.shape, weights)

  def buildStateful[E: Floating](
      input: Expr[E],
      output: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
    val (result, nextState) = model.buildStateful(input, weights, state)
    val loss = lossF.build(result, output) plus model.penalty(input.shape, weights)
    (loss, nextState)
  }

  def loss[E: Floating]: (Expr[E], Expr[E], Seq[Expr[E]]) => Expr[E] =
    (input, output, weights) => build(input, output, weights)

  def lossStateful[E: Floating]
      : (Expr[E], Expr[E], Seq[Expr[E]], Seq[Expr[E]]) => (Expr[E], Seq[Expr[E]]) =
    (input, output, weights, state) => buildStateful(input, output, weights, state)

  def grad[E: Floating]: (Expr[E], Expr[E], Seq[Expr[E]]) => Seq[Expr[E]] =
    (input, output, weights) => build(input, output, weights).grad(weights).returns[E]

  def gradStateful[E: Floating]
      : (Expr[E], Expr[E], Seq[Expr[E]], Seq[Expr[E]]) => (Seq[Expr[E]], Seq[Expr[E]]) =
    (input, output, weights, state) => {
      val (loss, nextState) = buildStateful(input, output, weights, state)
      val grad = loss.grad(weights).returns[E]
      (grad, nextState)
    }

  def trained[E: Floating](weights: Seq[Tensor[E]]) = new TrainedModel(this, weights)

  def displayLoss[E: Floating](input: Shape, dir: String = ""): Unit = {
    build(
      placeholder[E](input),
      placeholder[E](model.outputShape(input)),
      model.weightsShapes(input).map(s => placeholder[E](s)))
      .as("loss")
      .display(dir)
  }

  def displayGrad[E: Floating](input: Shape, dir: String = ""): Unit = {
    grad[E].apply(
      placeholder[E](input),
      placeholder[E](model.outputShape(input)),
      model.weightsShapes(input).map(s => placeholder[E](s)))
      .zipWithIndex
      .map { case (w, i) => w.as(s"loss_grad_${i}_layer") }
      .display(dir)
  }

  override def toString: String = s"$lossF($model)"
}

class TrainedModel[E: Floating](val lossModel: LossModel, val weights: Seq[Tensor[E]]) {

  def buildResult(input: Expr[E]): Expr[E] = lossModel.model.build(input, weights.map(_.const))

  def buildResultStateful(input: Expr[E], state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) =
    lossModel.model.buildStateful(input, weights.map(_.const), state)

  def result: Expr[E] => Expr[E] = (input: Expr[E]) => buildResult(input)

  def resultStateful: (Expr[E], Seq[Expr[E]]) => (Expr[E], Seq[Expr[E]]) =
    (input: Expr[E], state: Seq[Expr[E]]) => buildResultStateful(input, state)

  def buildLoss(input: Expr[E], output: Expr[E]): Expr[E] =
    lossModel.build(input, output, weights.map(_.const))

  def buildLossStateful(
      input: Expr[E],
      output: Expr[E],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) =
    lossModel.buildStateful(input, output, weights.map(_.const), state)

  def loss: (Expr[E], Expr[E]) => Expr[E] = (input, output) => buildLoss(input, output)

  def lossStateful: (Expr[E], Expr[E], Seq[Expr[E]]) => (Expr[E], Seq[Expr[E]]) =
    (input, output, state) => buildLossStateful(input, output, state)

  def outputShape(input: Shape): Shape = lossModel.model.outputShape(input)
}
