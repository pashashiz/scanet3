package scanet.models.layer

import scanet.core.{Expr, Floating, Shape}
import scanet.math.syntax._
import scanet.models.LayerInfo

import scala.collection.immutable
import scala.collection.immutable.Seq

/** Layer which composes 2 other layers
  *
  * @param left layer
  * @param right layer
  */
case class Composed(left: Layer, right: Layer) extends Layer {

  override def build[E: Floating](x: Expr[E], weights: Seq[Expr[E]]) = {
    val (leftWeights, rightWeights) = split(weights)
    val leftOutput = left.build(x, leftWeights)
    right.build(leftOutput, rightWeights)
  }

  override def penalty[E: Floating](weights: Seq[Expr[E]]) = {
    val (leftWeights, rightWeights) = split(weights)
    left.penalty(leftWeights) plus right.penalty(rightWeights)
  }

  override def outputShape(input: Shape): Shape = right.outputShape(left.outputShape(input))

  override def weightsCount: Int = left.weightsCount + right.weightsCount

  override def weightsShapes(input: Shape): immutable.Seq[Shape] = {
    val leftShapes = left.weightsShapes(input)
    val rightShapes = right.weightsShapes(left.outputShape(input))
    leftShapes ++ rightShapes
  }

  override def initWeights[E: Floating](input: Shape): Seq[Expr[E]] = {
    val leftShapes = left.initWeights[E](input)
    val rightShapes = right.initWeights[E](left.outputShape(input))
    leftShapes ++ rightShapes
  }

  private def split[E: Floating](weights: Seq[Expr[E]]) =
    weights.splitAt(left.weightsCount)

  override def info(input: Shape): Seq[LayerInfo] = {
    val rightInput = left.outputShape(input)
    left.info(input) ++ right.info(rightInput)
  }

  override def toString: String = s"$left >> $right"
}
