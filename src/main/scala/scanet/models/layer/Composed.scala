package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.math.syntax._

import scala.collection.immutable.Seq

/** Layer which composes 2 other layers
  *
  * @param left layer
  * @param right layer
  */
case class Composed(left: Layer, right: Layer) extends Layer {

  override def buildStateful[E: Floating](
      input: Expr[E],
      weights: Seq[Expr[E]],
      state: Seq[Expr[E]]): (Expr[E], Seq[Expr[E]]) = {
    val (leftWeights, rightWeights) = weights.splitAt(left.weightsShapes(input.shape).size)
    val (leftState, rightState) = state.splitAt(left.stateShapes(input.shape).size)
    val (leftOutput, leftNewState) = left.buildStateful(input, leftWeights, leftState)
    val (rightOutput, rightNewState) = right.buildStateful(leftOutput, rightWeights, rightState)
    (rightOutput, leftNewState ++ rightNewState)
  }

  override def penalty[E: Floating](input: Shape, weights: Seq[Expr[E]]) = {
    val (leftWeights, rightWeights) = weights.splitAt(left.weightsShapes(input).size)
    left.penalty(input, leftWeights) plus right.penalty(left.outputShape(input), rightWeights)
  }

  override def outputShape(input: Shape): Shape = right.outputShape(left.outputShape(input))

  override def weightsShapes(input: Shape): Seq[Shape] = {
    val leftShapes = left.weightsShapes(input)
    val rightShapes = right.weightsShapes(left.outputShape(input))
    leftShapes ++ rightShapes
  }

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] = {
    val leftShapes = left.initWeights[E](input)
    val rightShapes = right.initWeights[E](left.outputShape(input))
    leftShapes ++ rightShapes
  }

  override def stateShapes(input: Shape): Seq[Shape] = {
    val leftShapes = left.stateShapes(input)
    val rightShapes = right.stateShapes(left.outputShape(input))
    leftShapes ++ rightShapes
  }

  override def info(input: Shape): Seq[LayerInfo] = {
    val rightInput = left.outputShape(input)
    left.info(input) ++ right.info(rightInput)
  }

  override def toString: String = s"$left >> $right"

}
