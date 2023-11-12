package scanet.models

import scanet.core._
import scanet.math.syntax._
import scanet.models.layer.LayerInfo
import scanet.utils.{Bytes, Tabulator}

import scala.collection.immutable.Seq

abstract class Model extends Serializable {

  def name: String = getClass.getSimpleName.replace("$", "")

  /** Model params, both trainable and non-trainable (model state)
    * @param input input shape
    * @return param definitions
    */
  def params(input: Shape): Params[ParamDef]

  /** Build a model
    *
    * @param input training set, where first dimension equals to number of samples (batch size)
    * @param params initialized or calculated model params
    * @return tuple where the first element is model output and second is changed params
    */
  def build[E: Floating](input: Expr[E], params: Params[Expr[E]]): (Expr[E], Params[Expr[E]])

  /** Additional model penalty to be added to the loss
    *
    * @param params initialized or calculated model params
    * @return penalty
    */
  def penalty[E: Floating](params: Params[Expr[E]]): Expr[E]

  def result[E: Floating]: (Expr[E], Params[Expr[E]]) => Expr[E] =
    (input, params) => build(input, params)._1

  def resultStateful[E: Floating]: (Expr[E], Params[Expr[E]]) => (Expr[E], Params[Expr[E]]) =
    (input, params) => build(input, params)

  def outputShape(input: Shape): Shape

  def trainable: Boolean
  def makeTrainable(trainable: Boolean): Model
  def freeze: Model = makeTrainable(false)
  def unfreeze: Model = makeTrainable(true)

  def withLoss(loss: Loss): LossModel = LossModel(this, loss)

  private def makeGraph[E: Floating](input: Shape): Expr[E] =
    build(
      input = placeholder[E](input),
      params = params(input).mapValues(paramDef => placeholder[E](paramDef.shape)))
      ._1

  def displayResult[E: Floating](input: Shape, dir: String = ""): Unit =
    makeGraph[E](input).as("result").display(dir)

  def printResult[E: Floating](input: Shape): Unit =
    println(makeGraph[E](input).as("result").toString)

  def info(input: Shape): Seq[LayerInfo] = {
    val (weights, state) = params(input).partitionValues(_.trainable)
    Seq(LayerInfo(
      toString,
      weights.values.map(_.shape).toList,
      state.values.map(_.shape).toList,
      outputShape(input)))
  }

  def describe[E: Floating](input: Shape): String = {
    val layersInfo = info(input)
    val layers = (LayerInfo("Input", Seq.empty, Seq.empty, input) +: layersInfo).map(_.toRow)
    val table =
      Tabulator.format(Seq("name", "weights", "weights params", "state params", "output") +: layers)
    val weightTotal = layersInfo.map(info => info.weightsTotal).sum
    val weightSize = Bytes.formatSize(TensorType[E].codec.sizeOf(weightTotal))
    val stateTotal = layersInfo.map(info => info.stateTotal).sum
    val stateSize = Bytes.formatSize(TensorType[E].codec.sizeOf(stateTotal))
    s"$table\nTotal weight params: $weightTotal ($weightSize), state params: $stateTotal ($stateSize)"
  }
}

case class LossModel(model: Model, lossF: Loss) extends Serializable {

  def build[E: Floating](
      input: Expr[E],
      output: Expr[E],
      params: Params[Expr[E]]): Expr[E] =
    buildStateful(input, output, params)._1

  def buildStateful[E: Floating](
      input: Expr[E],
      output: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    val (result, nextParams) = model.build(input, params)
    val loss = lossF.build(result, output) plus model.penalty(params)
    (loss, nextParams)
  }

  def loss[E: Floating]: (Expr[E], Expr[E], Params[Expr[E]]) => Expr[E] =
    (input, output, params) => buildStateful(input, output, params)._1

  def lossStateful[E: Floating]: (Expr[E], Expr[E], Params[Expr[E]]) => (Expr[E], Params[Expr[E]]) =
    (input, output, params) => buildStateful(input, output, params)

  def grad[E: Floating]: (Expr[E], Expr[E], Params[Expr[E]]) => Params[Expr[E]] =
    (input, output, weights) => {
      val loss = build(input, output, weights)
      loss.grad(weights).returns[E]
    }

  def gradStateful[E: Floating]
      : (Expr[E], Expr[E], Params[Expr[E]], Params[Expr[E]]) => (Params[Expr[E]], Params[Expr[E]]) =
    (input, output, weights, state) => {
      val (loss, nextState) = buildStateful(input, output, weights ++ state)
      val grad = loss.grad(weights).returns[E]
      (grad, nextState)
    }

  def trainable: Boolean = model.trainable
  def makeTrainable(trainable: Boolean): LossModel = copy(model = model.makeTrainable(trainable))
  def freeze: LossModel = makeTrainable(false)
  def unfreeze: LossModel = makeTrainable(true)

  def trained[E: Floating](params: Params[Tensor[E]]): TrainedModel[E] =
    TrainedModel(this.freeze, params)

  def displayLoss[E: Floating](input: Shape, dir: String = ""): Unit = {
    val params = model.params(input)
    build(
      input = placeholder[E](input),
      output = placeholder[E](model.outputShape(input)),
      params = params.mapValues(paramDef => placeholder[E](paramDef.shape)))
      .as("loss")
      .display(dir)
  }

  def displayGrad[E: Floating](input: Shape, dir: String = ""): Unit = {
    val (weights, state) = model.params(input).partitionValues(_.trainable)
    val (grad, _) = gradStateful[E].apply(
      placeholder[E](input),
      placeholder[E](model.outputShape(input)),
      weights.mapValues(paramDef => placeholder[E](paramDef.shape)),
      state.mapValues(paramDef => placeholder[E](paramDef.shape)))
    grad.params
      .map { case (path, w) => (path, w.as(s"loss_grad_${path}_layer")) }
      .display(dir)
  }

  override def toString: String = s"$lossF($model)"
}

case class TrainedModel[E: Floating](lossModel: LossModel, params: Params[Tensor[E]]) {

  def buildResult(input: Expr[E]): Expr[E] =
    buildResultStateful(input)._1

  def buildResultStateful(input: Expr[E]): (Expr[E], Params[Expr[E]]) =
    lossModel.model.build(input, params.mapValues(_.const))

  def result: Expr[E] => Expr[E] = (input: Expr[E]) => buildResult(input)

  def resultStateful: Expr[E] => (Expr[E], Params[Expr[E]]) =
    (input: Expr[E]) => buildResultStateful(input)

  def buildLoss(input: Expr[E], output: Expr[E]): Expr[E] =
    buildLossStateful(input, output)._1

  def buildLossStateful(
      input: Expr[E],
      output: Expr[E]): (Expr[E], Params[Expr[E]]) =
    lossModel.buildStateful(input, output, params.mapValues(_.const))

  def loss: (Expr[E], Expr[E]) => Expr[E] = (input, output) => buildLoss(input, output)

  def lossStateful: (Expr[E], Expr[E]) => (Expr[E], Params[Expr[E]]) =
    (input, output) => buildLossStateful(input, output)

  def outputShape(input: Shape): Shape = lossModel.model.outputShape(input)

  def trainable: Boolean = lossModel.trainable
  def makeTrainable(trainable: Boolean): TrainedModel[E] =
    copy(lossModel = lossModel.makeTrainable(trainable))
  def freeze: TrainedModel[E] = makeTrainable(false)
  def unfreeze: TrainedModel[E] = makeTrainable(true)
}
