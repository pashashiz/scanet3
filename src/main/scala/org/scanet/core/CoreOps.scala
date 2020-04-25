package org.scanet.core

import simulacrum.typeclass
import scala.language.higherKinds
import org.scanet.core.TfType.syntax._

@typeclass trait ConstTensor[T[_]] {
  def const[A: TfType](t: T[A]): Output[A]
}

@typeclass trait ConstScalar[A] {
  def const(scalar: A): Output[A]
}

@typeclass trait CoreOps[A] {
  def as(op: A, label: String): A
}

object CoreOps {

  def const[A: TfType](tensor: Tensor[A], label: String): Output[A] =
    Output.name[A]("Const")
      .label(label)
      .shape(tensor.shape)
      .compileWithValue(tensor)
      .build

  trait Instances {

    implicit def tensorConst: ConstTensor[Tensor] = new ConstTensor[Tensor] {
      override def const[A: TfType](tensor: Tensor[A]): Output[A] = CoreOps.const(tensor, "Const")
    }

    implicit def floatConst: ConstScalar[Float] = (scalar: Float) => const(Tensor.scalar(scalar), "Const")
    implicit def doubleConst: ConstScalar[Double] = (scalar: Double) => const(Tensor.scalar(scalar), "Const")
    implicit def longConst: ConstScalar[Long] = (scalar: Long) => const(Tensor.scalar(scalar), "Const")
    implicit def intConst: ConstScalar[Int] = (scalar: Int) => const(Tensor.scalar(scalar), "Const")
    implicit def byteConst: ConstScalar[Byte] = (scalar: Byte) => const(Tensor.scalar(scalar), "Const")
    implicit def stringConst: ConstScalar[String] = (scalar: String) => const(Tensor.scalar(scalar), "Const")

    implicit def coreOps[A: TfType]: CoreOps[Output[A]] = (op: Output[A], label: String) => op.copy(label = label)

  }
  trait Syntax extends Instances with ConstTensor.ToConstTensorOps with ConstScalar.ToConstScalarOps with CoreOps.ToCoreOpsOps
  object syntax extends Syntax
}