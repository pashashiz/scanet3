package org.scanet.strings

import org.scanet.core.TensorType.syntax._
import org.scanet.core.{Output, TensorType, TfTypeString}
import simulacrum.typeclass

import scala.Ordering.Implicits._
import scala.language.higherKinds

@typeclass trait StringOps[F[_]] {

  def concat[A: TensorType : StringLike](left: F[A], right: F[A]): F[A]

  def length[A: TensorType : StringLike](op: F[A]): F[Int]

//  def split[A: TensorType : StringLike](op: F[A], sep: F[String]): F[String]
}

trait StringLike[S]

object StringOps {

  trait Instances {
    implicit def outputStringOps: StringOps[Output] = new OutputStringOps

    implicit def stringLikeInst: StringLike[String] = new StringLike[String] with TfTypeString
  }

  trait Syntax extends Instances with StringOps.ToStringOpsOps {

    def join(sep: String, ops: Output[String]*): Output[String] = {
      require(ops.zip(ops.tail).forall({ case (o1, o2) => o1.broadcastableAny(o2) }))
      Output.name[String]("StringJoin")
        .shape(ops.map(_.shape).max)
        .inputs(ops: _*)
        .compileWithAttr("separator", sep)
        .compileWithInputList
        .build
    }
  }

  object syntax extends Syntax

}

class OutputStringOps extends StringOps[Output] {

  override def concat[A: TensorType : StringLike](left: Output[A], right: Output[A]): Output[A] = {
    require(left.shape.broadcastableAny(right.shape))
    Output.name[A]("StringJoin")
      .shape(left.shape max right.shape)
      .inputs(left, right)
      .compileWithInputList
      .build
  }

  override def length[A: TensorType : StringLike](op: Output[A]): Output[Int] = {
    Output.name[Int]("StringLength")
      .shape(op.shape)
      .inputs(op)
      .compileWithAllInputs
      .build
  }

//  override def split[A: TensorType : StringLike](op: Output[A], sep: Output[String]): Output[String] = {
//    Output.name[String]("StringSplitV2")
//      .shape(op.shape) // TODO: this is wrong, result is SparseTensor with different shape
//      .inputs(op, sep)
//      .compileWithAllInputs
//      .build
//  }
}