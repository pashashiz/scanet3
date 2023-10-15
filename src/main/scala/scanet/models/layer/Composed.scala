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

  override def params_(input: Shape): Params[ParamDef] = {
    // todo: flatten
    val leftParams = left.params_(input).prependPath("l")
    val rightParams = right.params_(left.outputShape(input)).prependPath("r")
    leftParams ++ rightParams
  }

  override def build_[E: Floating](
      input: Expr[E],
      params: Params[Expr[E]]): (Expr[E], Params[Expr[E]]) = {
    val leftParams = params.children("l")
    val rightParams = params.children("r")
    val (leftOutput, leftState) = left.build_(input, leftParams)
    val (rightOutput, rightState) = right.build_(leftOutput, rightParams)
    (rightOutput, leftState.prependPath("l") ++ rightState.prependPath("r"))
  }

  override def penalty_[E: Floating](input: Shape, params: Params[Expr[E]]): Expr[E] = {
    val leftParams = params.children("l")
    val rightParams = params.children("r")
    left.penalty_(input, leftParams) plus right.penalty_(left.outputShape(input), rightParams)
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
