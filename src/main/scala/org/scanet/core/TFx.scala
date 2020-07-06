package org.scanet.core

import org.scanet.math.syntax.placeholder
import org.tensorflow.{Tensor => NativeTensor}
import simulacrum.typeclass


@typeclass trait SeqLike[F[_]] {
  def unit[P](seq: Seq[P]): F[P]
  def asSeq[P](arg: F[P]): Seq[P]
}

@typeclass trait SessionInputX[O] {
  def toInput(value: O): Seq[Output[_]]
}

@typeclass trait SessionOutputX[T] {
  def fromOutput(tensors: Seq[NativeTensor[_]]): T
}

class TFx1[Arg1[_]: SeqLike, P1: TensorType, In: SessionInputX, Out: SessionOutputX]
(val builder: Arg1[Shape] => (Seq[Output[P1]], In)) {
  def compile(session: Session): Arg1[Tensor[P1]] => Out = {
    a1 => {
      import TFx.syntax._
      val tensors = a1.asSeq
      val (p1, out) = builder(SeqLike[Arg1].unit(tensors.map(_.shape)))
      session.runner.feed(p1 zip tensors:_*).evalXX[In, Out](out)
    }
  }
}

object TFx1 {

  case class TFx1Builder[Arg1[_]: SeqLike, P1: TensorType, In: SessionInputX](builder: Arg1[Output[P1]] => In) {
    def returns[Out: SessionOutputX]: TFx1[Arg1, P1, In, Out] = new TFx1[Arg1, P1, In, Out](shapes => {
      import TFx.syntax._
      val placeholders = shapes.asSeq.map(shape => placeholder[P1](shape))
      val out = builder(SeqLike[Arg1].unit(placeholders))
      (placeholders, out)
    })
  }

  def apply[Arg1[_]: SeqLike, P1: TensorType, In: SessionInputX](builder: Arg1[Output[P1]] => In): TFx1Builder[Arg1, P1, In] = TFx1Builder(builder)
}

object TFx {

  trait Instances {

    implicit def seqIsArg: SeqLike[Seq] = new SeqLike[Seq] {
      override def unit[P](seq: Seq[P]): Seq[P] = seq
      override def asSeq[P](arg: Seq[P]): Seq[P] = arg
    }

    implicit def idIsArg: SeqLike[Id] = new SeqLike[Id] {
      override def unit[P](seq: Seq[P]): Id[P] = seq.head
      override def asSeq[P](arg: Id[P]): Seq[P] = Seq(arg)
    }

    implicit def singleOutputIsSessionInputX[SIn1[_]: SeqLike, A: TensorType]: SessionInputX[SIn1[Output[A]]] = {
      import SeqLike.ops._
      (out: SIn1[Output[A]]) => {
        import org.scanet.core.syntax._
        val outs = out.asSeq
        Tensor.vector(outs.size).const +: outs
      }
    }

    implicit def singleTensorIsSessionOutputX[SOut1[_]: SeqLike, A: TensorType]: SessionOutputX[SOut1[Tensor[A]]] = {
      (tensors: Seq[NativeTensor[_]]) => {
        import org.scanet.core.syntax._
        val size = Tensor.apply[Int](tensors(0).asInstanceOf[NativeTensor[Int]]).slice(0).toScalar
        val converted = tensors.slice(1, size + 1).map(nt => Tensor.apply[A](nt.asInstanceOf[NativeTensor[A]]))
        SeqLike[SOut1].unit(converted)
      }
    }
  }

  trait Syntax extends Instances with SeqLike.ToSeqLikeOps with SessionInputX.ToSessionInputXOps with SessionOutputX.ToSessionOutputXOps

  object syntax extends Syntax
}
