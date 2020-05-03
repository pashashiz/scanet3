package org.scanet.core

import org.scanet.math.Logical
import org.scanet.math.Numeric
import org.scanet.strings.Textual

class ConstOp[A: TensorType](val tensor: Tensor[A], val zero: Option[A], val one: Option[A]) {
   def const: Output[A] = ConstOp.buildConst(tensor, zero, one)
}

object ConstOp {

  def buildConst[A: TensorType](tensor: Tensor[A], zero: Option[A], one: Option[A]): Output[A] =
    Output.name[A]("Const")
      .shape(tensor.shape)
      .compileWithValue(tensor)
      .gradOpt(ctx => {
        (if (ctx.current == ctx.variable) one else zero)
          .map(v => buildConst(Tensor.fill(tensor.shape)(v), zero, one))
      })
      .build

  trait Syntax {

    implicit def numericScalarIsConstOp[A: TensorType: Numeric](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value), Some(Numeric[A].zero),  Some(Numeric[A].one))

    implicit def logicalScalarIsConstOp[A: TensorType: Logical](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value), None, None)

    implicit def textualScalarIsConstOp[A: TensorType: Textual](value: A): ConstOp[A] =
      new ConstOp(Tensor.scalar(value), None, None)

    implicit def numericTensorIsConstOp[A: TensorType: Numeric](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor, Some(Numeric[A].zero), Some(Numeric[A].one))

    implicit def logicalTensorIsConstOp[A: TensorType: Logical](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor, None, None)

    implicit def textualTensorIsConstOp[A: TensorType: Textual](tensor: Tensor[A]): ConstOp[A] =
      new ConstOp(tensor, None, None)
  }

  object syntax extends Syntax
}