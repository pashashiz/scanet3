package scanet.models.layer
import scanet.core.{Expr, Floating, Shape}
import scanet.models.Activation.{Sigmoid, Tanh}
import scanet.models.Initializer.{GlorotUniform, Ones, Orthogonal, Zeros}
import scanet.models.Regularization.Zero
import scanet.models.{Activation, Initializer, Regularization}
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
case class RNN(cell: Layer, returnSequence: Boolean = false) extends Layer {

  override def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
    require(input.rank == 3, "RNN requires input to have Shape(batch, time, features)")
    // input: (batch, time, features) -> stepsInput: (time, batch, features)
    val stepsInput = input.transpose(Seq(1, 0) ++ (2 until input.rank))
    val timeSteps = stepsInput.shape(0)
    @tailrec
    def stackCells(
        step: Int,
        outputs: Seq[Expr[E]],
        state: Seq[Expr[E]]): (Seq[Expr[E]], Seq[Expr[E]]) = {
      val (output, outputState) = cell.buildStateful(stepsInput.slice(step), weights, state)
      if (step < timeSteps - 1) {
        stackCells(step + 1, outputs :+ output, outputState)
      } else {
        (outputs :+ output, outputState)
      }
    }
    val (outputs, lastOutputState) = stackCells(0, Seq.empty, state)
    val output =
      if (returnSequence) joinAlong(outputs, 1).reshape(outputShape(input.shape))
      else outputs.last
    (output, lastOutputState)
  }

  private def dropTime(input: Shape): Shape = input.remove(1)

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E] =
    cell.penalty(dropTime(input), weights)

  override def outputShape(input: Shape): Shape = {
    val cellOutput = cell.outputShape(dropTime(input))
    if (returnSequence)
      cellOutput.insert(1, input(1))
    else
      cell.outputShape(cellOutput)
  }

  override def weightsShapes(input: Shape): Seq[Shape] = cell.weightsShapes(dropTime(input))

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] =
    cell.initWeights(dropTime(input))

  override def stateShapes(input: Shape): Seq[Shape] = cell.stateShapes(dropTime(input))
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
      returnSequence: Boolean = false): RNN =
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
      returnSequence)
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
}

case class SimpleRNNCell(
    units: Int,
    kernelInitializer: Initializer,
    recurrentInitializer: Initializer,
    kernelReg: Regularization,
    recurrentReg: Regularization)
    extends Layer {

  override def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
    require(input.rank >= 2, "SimpleRNNCell requires input Seq(batch, features)")
    require(weights.size == 2, "SimpleRNNCell requires weights Seq(kernel, recurrent)")
    require(state.size == 1, "SimpleRNNCell requires single state")
    val kernel = weights.head
    val recurrent = weights(1)
    val result = (input matmul kernel) + (state.head matmul recurrent)
    (result, Seq(result))
  }

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E] =
    kernelReg.build(weights.head) + recurrentReg.build(weights(1))

  override def outputShape(input: Shape): Shape = Shape(input.head, units)

  override def weightsShapes(input: Shape): Seq[Shape] = {
    val wx = Shape(input(1), units) // (features, units)
    val wh = Shape(units, units) // (units, units)
    Seq(wx, wh)
  }

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] = {
    val Seq(wx, wh) = weightsShapes(input)
    Seq(kernelInitializer.build[E](wx), recurrentInitializer.build[E](wh))
  }

  override def stateShapes(input: Shape): Seq[Shape] = Seq(outputShape(input))
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
      returnSequence: Boolean = false): RNN =
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
      returnSequence)
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

  private def cell(activation: Activation, useBias: Initializer) =
    SimpleRNNCell(
      units,
      activation,
      bias,
      kernelInitializer,
      recurrentInitializer,
      useBias,
      kernelReg,
      recurrentReg,
      biasReg)

  private val fCell = cell(recurrentActivation, useBias = biasForgetInitializer) // forget
  private val iCell = cell(recurrentActivation, useBias = biasInitializer) // input
  private val gCell = cell(activation, useBias = biasInitializer) // gate
  private val oCell = cell(recurrentActivation, useBias = biasInitializer) // output
  private val cells = Seq(fCell, iCell, gCell, oCell)

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
  override def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
    require(input.rank >= 2, "LSTMCell requires input Seq(batch, features)")
    require(
      weights.size == 8 + (if (bias) 4 else 0),
      "LSTMCell requires weights (Seq(kernel, recurrent, bias?) * 4")
    require(state.size == 2, "LSTMCell requires state Seq(c_prev, h_prev)")
    val Seq(cPrev, hPrev) = state
    val Seq(f, i, g, o) = cells.zip(unpackWeights(input.shape, weights)).map {
      case (cell, weights) => cell.buildStateful(input, weights, Seq(hPrev))._1
    }
    val c = cPrev * f + i * g
    val h = o * activation.build(c)
    (h, Seq(c, h))
  }

  private def unpackWeights[E](input: Shape, weights: Seq[Expr[E]]): Seq[Seq[Expr[E]]] = {
    val (_, unpacked) = cells.foldLeft((weights, Seq.empty[Seq[Expr[E]]])) {
      case ((weights, acc), cell) =>
        val size = cell.weightsShapes(input).size
        val (cellWeights, remainWeights) = weights.splitAt(size)
        (remainWeights, acc :+ cellWeights)
    }
    unpacked
  }

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E] =
    cells.zip(unpackWeights(input, weights)).foldLeft(Floating[E].zero.const) {
      case (sum, (cell, weights)) => sum + cell.penalty(input, weights)
    }

  override def outputShape(input: Shape): Shape =
    oCell.outputShape(input)

  override def weightsShapes(input: Shape): Seq[Shape] =
    cells.flatMap(_.weightsShapes(input))

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] =
    cells.flatMap(_.initWeights[E](input))

  override def stateShapes(input: Shape): Seq[Shape] =
    Seq.fill(2)(outputShape(input))
}
