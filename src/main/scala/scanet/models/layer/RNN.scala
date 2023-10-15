package scanet.models.layer

import scanet.core.{Expr, Floating, Params, Path, Shape}
import scanet.math.syntax.zeros
import scanet.models.Activation.{Sigmoid, Tanh}
import scanet.models.Aggregation.Avg
import scanet.models.Initializer.{GlorotUniform, Ones, Orthogonal, Zeros}
import scanet.models.Regularization.Zero
import scanet.models.{Activation, Initializer, ParamDef, Regularization}
import scanet.syntax._

import scala.annotation.tailrec
import scala.collection.immutable.Seq

/** RNN Layer
  *
  * Shapes:
  *  - input: (batch, time, features)
  *  - cell input: for each time (batch, features)
  *  - weights: same as cell
  *  - output: if `returnSequence=true` then (batch, time, units) else (batch, units)
  *
  * @param cell A RNN cell instance
  * @param returnSequence Whether to return the last output in the output sequence, or the full sequence.
  *                       To stack multiple layers set `returnSequence=true`
  */
case class RNN(cell: Layer, returnSequence: Boolean = false, stateful: Boolean = false)
    extends Layer {

  // todo: better params management

  override def params_(input: Shape): Params[ParamDef] = {
    val (weights, state) = paramsPartitioned(input)
    if (stateful) state ++ weights else weights
  }

  private def paramsPartitioned(input: Shape): (Params[ParamDef], Params[ParamDef]) =
    cell.params_(dropTime(input)).partitionValues(_.trainable)

  override def build_[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    require(input.rank == 3, "RNN requires input to have Shape(batch, time, features)")
    // input: (batch, time, features) -> stepsInput: (time, batch, features)
    val stepsInput = input.transpose(Seq(1, 0) ++ (2 until input.rank))
    val timeSteps = stepsInput.shape(0)
    val (weightParamsDef, stateParamsDef) = paramsPartitioned(input.shape)
    val weightParamsNames = weightParamsDef.params.keySet
    val (weightParams, stateParams) = params.params
      .partition { case (path, _) => weightParamsNames(path) }

    @tailrec
    def stackCells(
        step: Int,
        outputs: Seq[Expr[E]],
        state: Params[Expr[E]]): (Seq[Expr[E]], Params[Expr[E]]) = {
      val (output, outputState) = cell.build_(stepsInput.slice(step), state ++ Params(weightParams))
      if (step < timeSteps - 1) {
        stackCells(step + 1, outputs :+ output, outputState)
      } else {
        (outputs :+ output, outputState)
      }
    }

    val inputState =
      if (stateful) Params(stateParams) else stateParamsDef.mapValues(d => zeros[E](d.shape))
    val (outputs, lastOutputState) = stackCells(0, Seq.empty, inputState)
    val output =
      if (returnSequence) joinAlong(outputs, 1).reshape(outputShape(input.shape))
      else outputs.last
    (output, lastOutputState)
  }

  override def penalty_[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] =
    cell.penalty_(dropTime(input), params)

  private def dropTime(input: Shape): Shape = input.remove(1)

  override def outputShape(input: Shape): Shape = {
    val cellOutput = cell.outputShape(dropTime(input))
    if (returnSequence)
      cellOutput.insert(1, input(1))
    else
      cell.outputShape(cellOutput)
  }
}

object SimpleRNN {

  /** Simple RNN layer, where the output is to be fed back to input
    *
    * @param units Positive integer, dimensionality of the output space.
    * @param activation Activation function to use
    * @param bias Whether to add bias vector
    * @param kernelInitializer Initializer for the kernel weights matrix, used for the linear transformation of the inputs
    * @param recurrentInitializer Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state
    * @param biasInitializer Initializer for the bias vector
    * @param kernelReg Regularizer function applied to the kernel weights matrix
    * @param recurrentReg Regularizer function applied to the recurrent kernel weights matrix
    * @param biasReg Regularizer function applied to the bias vector
    * @param returnSequence Whether to return the last output in the output sequence, or the full sequence.
    *                       To stack multiple layers set `returnSequence=true`
    * @return Simple RNN layer
    */
  def apply(
      units: Int,
      activation: Activation = Tanh,
      bias: Boolean = true,
      kernelInitializer: Initializer = GlorotUniform(),
      recurrentInitializer: Initializer = Orthogonal(),
      biasInitializer: Initializer = Zeros,
      kernelReg: Regularization = Zero,
      recurrentReg: Regularization = Zero,
      biasReg: Regularization = Zero,
      returnSequence: Boolean = false,
      stateful: Boolean = false): RNN =
    RNN(
      SimpleRNNCell(
        units,
        activation,
        bias,
        kernelInitializer,
        recurrentInitializer,
        biasInitializer,
        kernelReg,
        recurrentReg,
        biasReg),
      returnSequence,
      stateful)
}

object SimpleRNNCell {

  def apply(
      units: Int,
      activation: Activation = Tanh,
      bias: Boolean = true,
      kernelInitializer: Initializer = GlorotUniform(),
      recurrentInitializer: Initializer = Orthogonal(),
      biasInitializer: Initializer = Zeros,
      kernelReg: Regularization = Zero,
      recurrentReg: Regularization = Zero,
      biasReg: Regularization = Zero): Layer = {
    require(units > 0, "RNN Cell should contain at least one unit")
    val cell =
      new SimpleRNNCell(units, kernelInitializer, recurrentInitializer, kernelReg, recurrentReg)
    cell ?>> (bias, Bias(units, biasReg, biasInitializer)) ?>> (activation.ni, activation.layer)
  }

  val Kernel: Path = "kernel_weights"
  val Recurrent: Path = "recurrent_weights"
  val State: Path = "state"
}

case class SimpleRNNCell(
    units: Int,
    kernelInitializer: Initializer,
    recurrentInitializer: Initializer,
    kernelReg: Regularization,
    recurrentReg: Regularization)
    extends Layer {
  import SimpleRNNCell._

  override def stateful: Boolean = true

  override def params_(input: Shape): Params[ParamDef] = Params(
    // (features, units)
    Kernel -> ParamDef(Shape(input(1), units), kernelInitializer, Some(Avg), trainable = true),
    // (units, units)
    Recurrent -> ParamDef(Shape(units, units), recurrentInitializer, Some(Avg), trainable = true),
    // state, keeps previous output
    State -> ParamDef(outputShape(input), Initializer.Zeros))

  override def build_[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    require(input.rank >= 2, "SimpleRNNCell requires input Seq(batch, features)")
    val result = (input matmul params(Kernel)) + (params(State) matmul params(Recurrent))
    (result, Params(State -> result))
  }

  override def penalty_[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] =
    kernelReg.build(params(Kernel)) + recurrentReg.build(params(Recurrent))

  override def outputShape(input: Shape): Shape = Shape(input.head, units)
}

object LSTM {

  /** Long Short-Term Memory layer - Hochreiter 1997
    *
    * @param units                Positive integer, dimensionality of the output space.
    * @param activation           Activation function to use
    * @param bias                 Whether to add bias vector
    * @param kernelInitializer    Initializer for the kernel weights matrix, used for the linear transformation of the inputs
    * @param recurrentInitializer Initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state
    * @param biasInitializer      Initializer for the bias vector
    * @param kernelReg            Regularizer function applied to the kernel weights matrix
    * @param recurrentReg         Regularizer function applied to the recurrent kernel weights matrix
    * @param biasReg              Regularizer function applied to the bias vector
    * @param returnSequence       Whether to return the last output in the output sequence, or the full sequence.
    *                             To stack multiple layers set `returnSequence=true`
    */
  def apply(
      units: Int,
      activation: Activation = Tanh,
      recurrentActivation: Activation = Sigmoid,
      bias: Boolean = true,
      kernelInitializer: Initializer = GlorotUniform(),
      recurrentInitializer: Initializer = Orthogonal(),
      biasInitializer: Initializer = Zeros,
      biasForgetInitializer: Initializer = Ones,
      kernelReg: Regularization = Zero,
      recurrentReg: Regularization = Zero,
      biasReg: Regularization = Zero,
      returnSequence: Boolean = false,
      stateful: Boolean = false): RNN =
    RNN(
      LSTMCell(
        units,
        activation,
        recurrentActivation,
        bias,
        kernelInitializer,
        recurrentInitializer,
        biasInitializer,
        biasForgetInitializer,
        kernelReg,
        recurrentReg,
        biasReg),
      returnSequence,
      stateful)
  val CellState: Path = "cell_state"
  val HiddenState: Path = "hidden_state"
}

case class LSTMCell(
    units: Int,
    activation: Activation = Tanh,
    recurrentActivation: Activation = Sigmoid,
    bias: Boolean = true,
    kernelInitializer: Initializer = GlorotUniform(),
    recurrentInitializer: Initializer = Orthogonal(),
    biasInitializer: Initializer = Zeros,
    biasForgetInitializer: Initializer = Ones,
    kernelReg: Regularization = Zero,
    recurrentReg: Regularization = Zero,
    biasReg: Regularization = Zero)
    extends Layer {
  import LSTM._

  private def cell(path: Path, activation: Activation, useBias: Initializer) =
    path -> SimpleRNNCell(
      units,
      activation,
      bias,
      kernelInitializer,
      recurrentInitializer,
      useBias,
      kernelReg,
      recurrentReg,
      biasReg)

  private val fCell = cell("forget", recurrentActivation, useBias = biasForgetInitializer)
  private val iCell = cell("input", recurrentActivation, useBias = biasInitializer)
  private val gCell = cell("gate", activation, useBias = biasInitializer)
  private val oCell = cell("output", recurrentActivation, useBias = biasInitializer)
  private val cells = Seq(fCell, iCell, gCell, oCell).map(_._2)
  private val cells_ = Map(fCell, iCell, gCell, oCell)

  override def stateful: Boolean = true

  override def params_(input: Shape): Params[ParamDef] = {
    val weights = cells_
      .map {
        case (name, cell) =>
          val params = cell.params_(input)
          val onlyWeights = params.filterPaths(path => !path.endsWith(SimpleRNNCell.State))
          onlyWeights.prependPath(name)
      }
      .reduce(_ ++ _)
    val states = Params(Seq(CellState, HiddenState)
      .map(path => path -> ParamDef(outputShape(input), Zeros)): _*)
    weights ++ states
  }

  /** Shapes:
    *  - input c t-1: (batch, features)
    *  - input h t-1: (batch, units)
    *  - input x: (batch, features)
    *  - for each gate [forget, input, gate, output]
    *    - kernel weights: (features, units)
    *    - recurrent weights: (units, units)
    *    - bias weights: (units)
    *  - output c: (batch, units)
    *  - output h: (batch, units)
    *  - output y: (batch, units)
    */
  override def build_[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    require(input.rank >= 2, "LSTMCell requires input Seq(batch, features)")
    val cPrev = params(CellState)
    val hPrev = params(HiddenState)
    val List(f, i, g, o) = cells_.toList.map {
      case (name, cell) =>
        val cellState = cell.params_(input.shape)
          .filter {
            case (path, param) =>
              path.endsWith(Params.State) && param.nonTrainable && param.shape == hPrev.shape
          }
          .mapValues(_ => hPrev)
        val cellParams = params.children(name) ++ cellState
        cell.build_(input, cellParams)._1
    }
    val c = cPrev * f + i * g
    val h = o * activation.build(c)
    (h, Params(CellState -> c, HiddenState -> h))
  }

  override def penalty_[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] =
    cells_.foldLeft(Floating[E].zero.const) {
      case (sum, (name, cell)) =>
        val cellParams = params.children(name)
        sum + cell.penalty_(input, cellParams)
    }

  override def outputShape(input: Shape): Shape =
    oCell._2.outputShape(input)
}
