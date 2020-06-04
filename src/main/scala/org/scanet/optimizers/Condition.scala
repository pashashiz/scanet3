package org.scanet.optimizers

import org.scanet.core.TensorType
import org.scanet.math.Numeric
import simulacrum.typeclass

case class Condition[W: Numeric: TensorType, R: Numeric: TensorType](f: Step[W, R] => Boolean) extends (Step[W, R] => Boolean) {
  override def apply(step: Step[W, R]): Boolean = f(step)
}

object Condition {

  def always[W: Numeric: TensorType, R: Numeric: TensorType]: Condition[W, R] = Condition(_ => true)

  def iterations[W: Numeric: TensorType, R: Numeric: TensorType](number: Int): Condition[W, R] = {
    Condition(step => {
      step.total % number == 0
    })
  }

  def epochs[W: Numeric: TensorType, R: Numeric: TensorType](number: Int): Condition[W, R] =
    Condition(step => {
      step.isLastIter && step.epoch % number == 0
    })

  trait Implicits {

    implicit def canBuildConditionFromInt: CanBuildConditionFrom[Int] = new CanBuildConditionFrom[Int] {
      override def iterations[W: Numeric : TensorType, R: Numeric : TensorType](a: Int): Condition[W, R] =
        Condition.iterations[W, R](a)
      override def epochs[W: Numeric : TensorType, R: Numeric : TensorType](a: Int): Condition[W, R] =
        Condition.epochs[W, R](a)
    }
  }

  trait Syntax extends Implicits with CanBuildConditionFrom.ToCanBuildConditionFromOps

  object syntax extends Syntax
}

@typeclass trait CanBuildConditionFrom[A] {
  def iterations[W: Numeric: TensorType, R: Numeric: TensorType](a: A): Condition[W, R]
  def epochs[W: Numeric: TensorType, R: Numeric: TensorType](a: A): Condition[W, R]
}

