package org.scanet.core

import org.scanet.math.Logical
import org.scanet.math.Numeric
import org.scanet.strings.Textual

class ConstOp[A: TensorType](val tensor: Tensor[A]) {
   def const: Output[A] = ConstOp.buildConst(tensor)
}

object ConstOp {

  def buildConst[A: TensorType](tensor: Tensor[A]): Output[A] =
    Output.name[A]("Const")
      .shape(tensor.shape)
      .compileWithValue(tensor)
      .localGrad[A](_ => Map())
      .build

  trait Syntax {

    implicit def numericScalarIsConstOp[A: TensorType: Numeric](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value))

    implicit def logicalScalarIsConstOp[A: TensorType: Logical](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value))

    implicit def textualScalarIsConstOp[A: TensorType: Textual](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value))

    implicit def numericTensorIsConstOp[A: TensorType: Numeric](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor)

    implicit def logicalTensorIsConstOp[A: TensorType: Logical](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor)

    implicit def textualTensorIsConstOp[A: TensorType: Textual](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor)
  }

  object syntax extends Syntax
}