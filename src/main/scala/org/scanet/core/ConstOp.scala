package org.scanet.core

class ConstOp[A: TensorType](val tensor: Tensor[A]) {
   def const: Output[A] = ConstOp.buildConst(tensor)
}

object ConstOp {

  def buildConst[A: TensorType](tensor: Tensor[A]): Output[A] =
    Output.name[A]("Const")
      .shape(tensor.shape)
      .compileWithValue(tensor)
      .localGrad(_ => List())
      .build

  trait Syntax {

    implicit def scalarIsConstOp[A: TensorType](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value))

    implicit def tensorIsConstOp[A: TensorType](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor)
  }

  object syntax extends Syntax
}