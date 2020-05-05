package org.scanet.math

import org.scanet.core.Output.GradContext
import org.scanet.core.{Output, TensorType}
import simulacrum.typeclass

@typeclass trait MathGradOp[F[_]] {
  def grad[A: TensorType: Numeric](current: F[A], withRespectTo: Output[_]): F[A]
}

object MathGradOp {

  trait Instances {
    implicit def outputIsMathGradOp: MathGradOp[Output] = new OutputIsMathGradOp
  }

  trait Syntax extends Instances with MathGradOp.ToMathGradOpOps

  object syntax extends Syntax
}

class OutputIsMathGradOp extends MathGradOp[Output] {
  override def grad[A: TensorType : Numeric](current: Output[A], withRespectTo: Output[_]): Output[A] =
    current.gradF(GradContext(current, withRespectTo)).get
}
