package org.scanet.models.nn

import org.scanet.core.{Output, OutputSeq, TensorType}
import org.scanet.math.{Floating, Numeric}

/**
 * Layer which composes 2 other layers
 *
 * @param left layer
 * @param right layer
 */
case class Composed(left: Layer, right: Layer) extends Layer {

  override def build[E: Numeric : Floating : TensorType](x: Output[E], weights: OutputSeq[E]) = {
    val (leftWeights, rightWeights) = weights.splitAt(left.shapes(0).size)
    val leftOutput = left.build(x, leftWeights)
    right.build(leftOutput, rightWeights)
  }

  override def outputs() = right.outputs()

  override def shapes(features: Int) = {
    val leftShapes = left.shapes(features)
    val rightOutputs = leftShapes.last.head
    val rightShapes = right.shapes(rightOutputs)
    leftShapes ++ rightShapes
  }
}
