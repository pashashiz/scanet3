package org.scanet.optimizers

import org.scanet.core.TensorType
import org.scanet.math.Numeric
import simulacrum.typeclass

case class Condition[A](f: A => Boolean) extends (A => Boolean) {
  override def apply(value: A): Boolean = f(value)
}

object Condition {

  def always[R: Numeric: TensorType]: Condition[StepContext[R]] = Condition(_ => true)

  def never[R: Numeric: TensorType]: Condition[StepContext[R]] = Condition(_ => false)

  def iterations[R: Numeric: TensorType](number: Int): Condition[StepContext[R]] = {
    Condition(ctx => {
      ctx.step.iter % number == 0
    })
  }

  def epochs[R: Numeric: TensorType](number: Int): Condition[StepContext[R]] =
    Condition(ctx => {
      ctx.step.epoch % number == 0
    })

  trait Implicits {

    implicit def canBuildConditionFromInt: CanBuildConditionFrom[Int] = new CanBuildConditionFrom[Int] {
      override def iterations[R: Numeric : TensorType](a: Int): Condition[StepContext[R]] =
        Condition.iterations[R](a)
      override def epochs[R: Numeric : TensorType](a: Int): Condition[StepContext[R]] =
        Condition.epochs[R](a)
    }
  }

  trait Syntax extends Implicits with CanBuildConditionFrom.ToCanBuildConditionFromOps

  object syntax extends Syntax
}

@typeclass trait CanBuildConditionFrom[A] {
  def iterations[R: Numeric: TensorType](a: A): Condition[StepContext[R]]
  def epochs[R: Numeric: TensorType](a: A): Condition[StepContext[R]]
}

