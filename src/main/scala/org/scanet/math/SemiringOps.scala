package org.scanet.math

import org.scanet.core.{Output, TfType}

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
      .label("Add")
      .shape(if (left.rank > rightOut.rank) left.shape else rightOut.shape)
      .inputs(left, rightOut)
      .compileWithAllInputs
      .build
  }

  override def multiply[B](left: Output[A], right: B)(implicit c: Convertible[B, Output[A]]): Output[A] = ???
}