package scanet.models.layer
import scanet.core.{Expr, Floating, Shape}
import scanet.models.Activation.{Sigmoid, Tanh}
import scanet.models.Initializer.{GlorotUniform, Orthogonal, Zeros}
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
  *  - output: same as cell
  *
  *  todo:
  *   - unlock output from all cells, not only the last one
  *   - allow stacking RNN layers
  *   - add unroll option, see https://www.tensorflow.org/api_docs/python/tf/while_loop
  *   - add state between batches (requires ordering)
  * @param cell A RNN cell instance
  */
case class RNN(cell: Layer) extends Layer {

  override def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
    require(input.rank == 3, "RNN requires input to have Shape(batch, time, features)")
    // input: (batch, time, features) -> stepsInput: (time, batch, features)
    val stepsInput = input.transpose(Seq(1, 0) ++ (2 until input.rank))
    val timeSteps = stepsInput.shape(0)
    @tailrec
    def stackCells(step: Int, state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
      val (output, nextState) = cell.buildStateful(stepsInput.slice(step), weights, state)
      if (step < timeSteps - 1) stackCells(step + 1, nextState)
      else (output, nextState)
    }
    stackCells(0, state)
  }

  private def dropTime(input: Shape): Shape = input.remove(1)

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]): Expr[E] =
    cell.penalty(dropTime(input), weights)

  override def outputShape(input: Shape): Shape = cell.outputShape(dropTime(input))

  override def weightsShapes(input: Shape): Seq[Shape] = cell.weightsShapes(dropTime(input))

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] =
    cell.initWeights(dropTime(input))

  override def stateShapes(input: Shape): Seq[Shape] = cell.stateShapes(dropTime(input))
}

object SimpleRNNCell {

  /** Simple RNN Cell, where the output is to be fed back to input
    *
    * Shapes:
    *  - input x: (batch, features)
    *  - input h t-1: (batch, units)
    *  - kernel weights: (features, units)
    *  - recurrent weights: (units, units)
    *  - bias weights: (units)
    *  - output: (batch, units)
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
    * @return [[SimpleRNNCell]] >> [[Bias]] >> [[Activation]]
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

case class LSTMCell(
    units: Int,
    activation: Activation = Tanh,
    recurrentActivation: Activation = Sigmoid,
    bias: Boolean = true,
    kernelInitializer: Initializer = GlorotUniform(),
    recurrentInitializer: Initializer = Orthogonal(),
    biasInitializer: Initializer = Zeros,
    kernelReg: Regularization = Zero,
    recurrentReg: Regularization = Zero,
    biasReg: Regularization = Zero)
    extends Layer {

  private def cell(activation: Activation) =
    SimpleRNNCell(
      units,
      activation,
      bias,
      kernelInitializer,
      recurrentInitializer,
      biasInitializer,
      kernelReg,
      recurrentReg,
      biasReg)

  private val fCell = cell(recurrentActivation) // forget
  private val iCell = cell(recurrentActivation) // input
  private val gCell = cell(activation) // gate
  private val oCell = cell(recurrentActivation) // output
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
