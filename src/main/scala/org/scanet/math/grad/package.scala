package org.scanet.math.grad

import org.scanet.core.{Expr, Node, Shape, TensorType}
import org.scanet.math._
import org.scanet.math.alg.kernels.syntax._
import scala.collection.immutable.Seq

class GradCalcN[A: TensorType: Numeric, B: TensorType: Numeric](
    out: Expr[A],
    withRespectTo: Seq[Expr[B]]) {
  def returns[R: Floating: Numeric: TensorType]: Seq[Expr[R]] =
    withRespectTo.map(r => new GradCalc[A, B](out, r).returns[R])
}

class GradCalc[A: TensorType: Numeric, B: TensorType: Numeric](
    out: Expr[A],
    withRespectTo: Expr[B]) {
  def returns[R: Floating: Numeric: TensorType]: Expr[R] = {
    require(
      out.shape.isScalar,
      "gradient is supported on scalars only, " +
      "reduce the output with sum() or other operation")
    val graph = out.asGraph
    val leaf = graph.find(withRespectTo.toString)
    require(
      leaf.isDefined,
      s"cannot find a gradient with respect to $withRespectTo " +
      s"cause that input is not a part of the computation graph")
    def gradRec(node: Node[Expr[_]]): Expr[R] = {
      if (node.isRoot) {
        ones[R](Shape())
      } else {
        val grads = node.outputs.map(edge => {
          val parent = edge.to
          val parentGrad = gradRec(parent)
          parent.value.localGrad(edge.index, parentGrad)
        })
        plus(grads.toList)
      }
    }
    leaf.map(gradRec).get
  }
}

class GradOps[A: TensorType: Numeric](expr: Expr[A]) {

  def grad[B: TensorType: Numeric](withRespectTo: Expr[B]): GradCalc[A, B] =
    new GradCalc[A, B](expr, withRespectTo)

  def grad[B: TensorType: Numeric](withRespectTo: Seq[Expr[B]]): GradCalcN[A, B] =
    new GradCalcN[A, B](expr, withRespectTo)
}

trait GradSyntax {
  implicit def toGradOps[A: TensorType: Numeric](expr: Expr[A]): GradOps[A] =
    new GradOps[A](expr)
}

object syntax extends GradSyntax
