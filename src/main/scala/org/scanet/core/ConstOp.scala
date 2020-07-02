package org.scanet.core

import org.scanet
import org.scanet.core.Output.Grad
import org.scanet.math.Floating


class ConstOp[A: TensorType](val tensor: Tensor[A]) {
   def const: Output[A] = ConstOp.buildConst(tensor)
}

object ConstOp {

  def buildConst[A: TensorType](tensor: Tensor[A]): Output[A] =
    Output.name[A]("Const")
      .shape(tensor.shape)
      .compileWithValue(tensor)
      .localGrad(new Grad[A] {
        override def calc[R: scanet.math.Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] =
          List()
      })
      .build

  trait Syntax {

    implicit def scalarIsConstOp[A: TensorType](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value))

    implicit def tensorIsConstOp[A: TensorType](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor)
  }

  object syntax extends Syntax
}