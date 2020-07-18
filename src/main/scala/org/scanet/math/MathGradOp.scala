package org.scanet.math

import org.scanet.core.{Node, Output, Shape, TensorType}
import org.scanet.math.MathBaseOp.syntax._
import simulacrum.typeclass

@typeclass trait MathGradOp[F[_]] {

  def grad[A: TensorType : Numeric, B: TensorType : Numeric](
    current: F[A], withRespectTo: Output[B]): GradCalc[A, B]

  def grad[A: TensorType : Numeric, B: TensorType : Numeric](
    current: F[A], withRespectTo: Seq[Output[B]]): GradCalcN[A, B]
}

object MathGradOp {

  trait Instances {
    implicit def outputIsMathGradOp: MathGradOp[Output] = new OutputIsMathGradOp
  }

  trait Syntax extends Instances with MathGradOp.ToMathGradOpOps

  object syntax extends Syntax
}

class OutputIsMathGradOp extends MathGradOp[Output] {

  override def grad[A: TensorType : Numeric, B: TensorType : Numeric](
       out: Output[A], withRespectTo: Output[B]): GradCalc[A, B] = new GradCalc[A, B](out, withRespectTo)

  override def grad[A: TensorType : Numeric, B: TensorType : Numeric](out: Output[A], withRespectTo: Seq[Output[B]]) =
    new GradCalcN[A, B](out, withRespectTo)
}

class GradCalcN[A: TensorType : Numeric, B: TensorType : Numeric](out: Output[A], withRespectTo: Seq[Output[B]]) {
  def returns[R: Floating: Numeric: TensorType]: Seq[Output[R]] =
    withRespectTo.map(r => new GradCalc[A, B](out, r).returns[R])
}

class GradCalc[A: TensorType : Numeric, B: TensorType : Numeric](out: Output[A], withRespectTo: Output[B]) {
  def returns[R: Floating: Numeric: TensorType]: Output[R] = {
    require(out.shape.isScalar, "gradient is supported on scalars only, " +
      "reduce the output with sum() or other operation")
    val graph = out.asGraph
    val leaf = graph.find(withRespectTo.toString)
    require(leaf.isDefined, s"cannot find a gradient with respect to $withRespectTo " +
      s"cause that input is not a part of the computation graph")
    def gradRec(node: Node[Output[_]]): Output[R] = {
      if (node.isRoot) {
        ones[R](Shape())
      } else {
        val grads = node.outputs.map(edge => {
          val parent = edge.to
          val parentGrad = gradRec(parent)
          parent.value.localGrad(edge.index, parentGrad)
        })
        plus(grads.toList: _*)
      }
    }
    leaf.map(gradRec).get
  }
}