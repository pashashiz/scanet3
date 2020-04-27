package org.scanet.math

import org.scanet.core.{Output, Shape, TfType}
import org.scanet.core.CoreOps.syntax._

object SemiringOps {

  trait Instances {
    implicit def semiringOps[A: TfType: Semiring]: Semiring[Output[A]] = new SemiringOps[A]
  }

  trait Syntax extends Instances with Semiring.ToSemiringOps

  object syntax extends Syntax
}

class SemiringOps[A: TfType: Semiring] extends Semiring[Output[A]] {

  override def plus[B](left: Output[A], right: B)(implicit c: Convertible[B, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    require(left.shape.endsWith(rightOut.shape) || rightOut.shape.endsWith(left.shape) ,
      s"tensors with shapes ${left.shape} and ${rightOut.shape} cannot be added, " +
        "one of the tensors should have shape which includes the other")
    Output.name("Add")
      .shape(if (left.rank > rightOut.rank) left.shape else rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def multiply[B](left: Output[A], right: B)(implicit c: Convertible[B, Output[A]]): Output[A] = {
    val rightOut: Output[A] = c.convert(right)
    val leftAdjusted = left.reshape(left.shape.alignLeft(2, using = 1))
    val rightAdjusted = rightOut.reshape(rightOut.shape.alignLeft(2, using = 1))
    require(leftAdjusted.rank == 2 && rightAdjusted.rank == 2,
      s"rank cannot be > 2 but got tensors with shapes ${leftAdjusted.shape} * ${rightAdjusted.shape}")
    require(leftAdjusted.shape.last == rightAdjusted.shape.head,
      s"cannot multiply tensors with shapes ${leftAdjusted.shape} * ${rightAdjusted.shape}")
    val resultShape = Shape(leftAdjusted.shape.head, rightAdjusted.shape.last)
    val result = Output.name("MatMul")
      .shape(resultShape)
      .inputs(leftAdjusted, rightAdjusted)
      .compileWithAllInputs
      .build
    // we need to prune additional adjusted dimensions added for scalars and vectors
    val adjusted = 2 - math.min(left.shape.rank, rightOut.shape.rank)
    result.reshape(resultShape.prune(adjusted))
  }
}
