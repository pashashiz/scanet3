package org.scanet.math

import org.scanet.core.{Node, Output, Shape, Tensor, TensorType}
import org.scanet.core.syntax._
import org.scanet.math.MathBaseOp.syntax._
import org.scanet.math.Numeric.syntax._
import simulacrum.typeclass

@typeclass trait MathGradOp[F[_]] {
  def grad[A: TensorType : Numeric, B: TensorType : Numeric](current: F[A], withRespectTo: Output[B]): F[Float]
}

object MathGradOp {

  trait Instances {
    implicit def outputIsMathGradOp: MathGradOp[Output] = new OutputIsMathGradOp
  }

  trait Syntax extends Instances with MathGradOp.ToMathGradOpOps

  object syntax extends Syntax
}

class OutputIsMathGradOp extends MathGradOp[Output] {
  override def grad[A: TensorType : Numeric, B: TensorType : Numeric](out: Output[A], withRespectTo: Output[B]): Output[Float] = {
    require(out.shape.isScalar, "gradient is supported on scalars only, " +
      "reduce the output with sum() or other operation")
    val graph = out.asGraph
    val leaf = graph.find(withRespectTo.id)
    require(leaf.isDefined, s"cannot find a gradient with respect to $withRespectTo " +
      s"cause that input is not a part of the computation graph")
    def gradRec(node: Node[Output[_]]): Output[Float] = {
      if (node.isRoot) {
        Tensor.ones[Float](Shape()).const
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
