package org.scanet.core

import simulacrum.typeclass
import org.scanet.core.TensorType.syntax._

@typeclass trait ConstTensor[T[_]] {
  def const[A: TensorType](t: T[A]): Output[A]
}

@typeclass trait ConstScalar[A] {
  def const(scalar: A): Output[A]
}

object Const {

  def apply[A: TensorType](tensor: Tensor[A], label: String): Output[A] =
    Output.name[A]("Const")
      .label(label)
      .shape(tensor.shape)
      .compileWithValue(tensor)
      .gradOpt(ctx => {
        val value =
          if (ctx.current == ctx.variable) TensorType[A].oneIfNumeric
          else TensorType[A].zeroIfNumeric
        value.map(v => Const(Tensor.fill(tensor.shape)(v), "Const"))
      })
      .build

  trait Instances {

    implicit def tensorConst: ConstTensor[Tensor] = new ConstTensor[Tensor] {
      override def const[A: TensorType](tensor: Tensor[A]): Output[A] = Const(tensor, "Const")
    }

    implicit def floatConst: ConstScalar[Float] = (scalar: Float) => Const(Tensor.scalar(scalar), "Const")
    implicit def doubleConst: ConstScalar[Double] = (scalar: Double) => Const(Tensor.scalar(scalar), "Const")
    implicit def longConst: ConstScalar[Long] = (scalar: Long) => Const(Tensor.scalar(scalar), "Const")
    implicit def intConst: ConstScalar[Int] = (scalar: Int) => Const(Tensor.scalar(scalar), "Const")
    implicit def byteConst: ConstScalar[Byte] = (scalar: Byte) => Const(Tensor.scalar(scalar), "Const")
    implicit def stringConst: ConstScalar[String] = (scalar: String) => Const(Tensor.scalar(scalar), "Const")

  }
  trait Syntax extends Instances with ConstTensor.ToConstTensorOps with ConstScalar.ToConstScalarOps
  object syntax extends Syntax

}
