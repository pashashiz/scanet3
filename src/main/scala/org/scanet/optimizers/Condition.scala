package org.scanet.optimizers

import org.scanet.core.TensorType
import org.scanet.math.Numeric
import simulacrum.typeclass

case class Condition[R: Numeric: TensorType](f: Step[R] => Boolean) extends (Step[R] => Boolean) {
  override def apply(step: Step[R]): Boolean = f(step)
}

object Condition {

  def always[R: Numeric: TensorType]: Condition[R] = Condition(_ => true)

  def never[R: Numeric: TensorType]: Condition[R] = Condition(_ => false)

  def iterations[R: Numeric: TensorType](number: Int): Condition[R] = {
    Condition(step => {
      step.iter % number == 0
    })
  }

  def epochs[R: Numeric: TensorType](number: Int): Condition[R] =
    Condition(step => {
      step.epoch % number == 0
    })

  trait Implicits {

    implicit def canBuildConditionFromInt: CanBuildConditionFrom[Int] = new CanBuildConditionFrom[Int] {
      override def iterations[R: Numeric : TensorType](a: Int): Condition[R] =
        Condition.iterations[R](a)
      override def epochs[R: Numeric : TensorType](a: Int): Condition[R] =
        Condition.epochs[R](a)
    }
  }

  trait Syntax extends Implicits with CanBuildConditionFrom.ToCanBuildConditionFromOps

  object syntax extends Syntax
}

@typeclass trait CanBuildConditionFrom[A] {
  def iterations[R: Numeric: TensorType](a: A): Condition[R]
  def epochs[R: Numeric: TensorType](a: A): Condition[R]
}

