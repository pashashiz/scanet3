package scanet.models.layer

import scanet.core.{Expr, Floating, Params, Shape}
import scanet.math.syntax._
import scanet.models.ParamDef

import scala.collection.immutable.Seq

/** Layer which composes 2 other layers
  *
  * @param left layer
  * @param right layer
  */
case class Composed(left: Layer, right: Layer) extends Layer {

  override def params(input: Shape): Params[ParamDef] = {
    // todo: flatten
    val leftParams = left.params(input).prependPath("l")
    val rightParams = right.params(left.outputShape(input)).prependPath("r")
    leftParams ++ rightParams
  }

  override def build[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    val leftParams = params.children("l")
    val rightParams = params.children("r")
    val (leftOutput, leftState) = left.build(input, leftParams)
    val (rightOutput, rightState) = right.build(leftOutput, rightParams)
    (rightOutput, leftState.prependPath("l") ++ rightState.prependPath("r"))
  }

  override def penalty[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] = {
    val leftParams = params.children("l")
    val rightParams = params.children("r")
    left.penalty(input, leftParams) plus right.penalty(left.outputShape(input), rightParams)
  }

  override def outputShape(input: Shape): Shape =
    right.outputShape(left.outputShape(input))

  override def stateful: Boolean =
    left.stateful || right.stateful

  override def info(input: Shape): Seq[LayerInfo] = {
    val rightInput = left.outputShape(input)
    left.info(input) ++ right.info(rightInput)
  }

  override def toString: String = s"$left >> $right"
}
