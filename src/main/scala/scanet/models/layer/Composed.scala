package scanet.models.layer

import scanet.core.{Expr, OutputSeq, TensorType}
import scanet.math.{Floating, Numeric}
import scanet.math.syntax._

/** Layer which composes 2 other layers
  *
  * @param left layer
  * @param right layer
  */
case class Composed(left: Layer, right: Layer) extends Layer {

  override def build[E: Numeric: Floating: TensorType](x: Expr[E], weights: OutputSeq[E]) = {
    val (leftWeights, rightWeights) = split(weights)
    val leftOutput = left.build(x, leftWeights)
    right.build(leftOutput, rightWeights)
  }

  override def penalty[E: Numeric: Floating: TensorType](weights: OutputSeq[E]) = {
    val (leftWeights, rightWeights) = split(weights)
    left.penalty(leftWeights) plus left.penalty(rightWeights)
  }

  override def outputs() = right.outputs()

  override def shapes(features: Int) = {
    val leftShapes = left.shapes(features)
    val rightOutputs = leftShapes.last.head
    val rightShapes = right.shapes(rightOutputs)
    leftShapes ++ rightShapes
  }

  private def split[E: Numeric: Floating: TensorType](weights: OutputSeq[E]) = {
    weights.splitAt(left.shapes(0).size)
  }
}
