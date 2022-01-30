package scanet.core

import org.tensorflow.RawTensor
import scanet.core.Layout._

import scala.collection.immutable.Seq

// Important while programming dependant types
// - use concrete types where type implementation is specified
// - when mixing with implicits pass via implicit parameter only and do not use context bounds
//   case type information gets lost there
trait Mat[In] {
  type Out
  def deconstructIn(in: In): (Layout, Seq[Expr[_]])
  def constructIn(shape: Layout, expr: Seq[Expr[_]]): In
  def deconstructOut(out: Out): (Layout, Seq[Tensor[_]])
  def constructOutRaw(shape: Layout, tensors: Seq[RawTensor]): Out
}

sealed trait Layout {
  def size: Int
}

object Layout {
  case class Leaf(size: Int) extends Layout
  case class Struct(children: Seq[Layout]) extends Layout {
    def apply(index: Int): Layout = children(index)
    def size: Int = children.map(_.size).sum
  }
}

class MatExpr[A: TensorType] extends Mat[Expr[A]] {
  override type Out = Tensor[A]
  override def deconstructIn(in: Expr[A]): (Layout, Seq[Expr[_]]) = (Leaf(1), Seq(in))
  override def constructIn(shape: Layout, expr: Seq[Expr[_]]): Expr[A] =
    expr.head.asInstanceOf[Expr[A]]
  override def deconstructOut(out: Tensor[A]): (Layout, Seq[Tensor[_]]) = (Leaf(1), Seq(out))
  override def constructOutRaw(shape: Layout, tensors: Seq[RawTensor]): Out =
    Tensor.wrap[A](tensors.head)
}

class MatSeqOfExpr[A: TensorType] extends Mat[Seq[Expr[A]]] {
  override type Out = Seq[Tensor[A]]
  override def deconstructIn(in: Seq[Expr[A]]): (Layout, Seq[Expr[_]]) = (Leaf(in.size), in)
  override def constructIn(shape: Layout, expr: Seq[Expr[_]]): Seq[Expr[A]] =
    expr.asInstanceOf[Seq[Expr[A]]]
  override def deconstructOut(out: Seq[Tensor[A]]): (Layout, Seq[Tensor[_]]) =
    (Leaf(out.size), out)
  override def constructOutRaw(shape: Layout, tensors: Seq[RawTensor]): Out =
    tensors.map(raw => Tensor.wrap[A](raw))
}

object Mat {

  trait AllSyntax {

    implicit def matExpr[A: TensorType]: MatExpr[A] = new MatExpr[A]

    implicit def matSeqOfExpr[A: TensorType]: MatSeqOfExpr[A] = new MatSeqOfExpr[A]

    // Note: we have to define anonymous class so dependant types would work
    // that is probably caused by the fact that we have to preserve an original
    // (m1.Out, m2.Out) which come from implicit scope
    implicit def matTuple2Expr[A1, A2](implicit m1: Mat[A1], m2: Mat[A2]) =
      new Mat[(A1, A2)] {
        override type Out = (m1.Out, m2.Out)
        override def deconstructIn(in: (A1, A2)): (Layout, Seq[Expr[_]]) = {
          val (shape1, expr1) = m1.deconstructIn(in._1)
          val (shape2, expr2) = m2.deconstructIn(in._2)
          (Struct(Seq(shape1, shape2)), expr1 ++ expr2)
        }
        override def constructIn(shape: Layout, expr: Seq[Expr[_]]): (A1, A2) = {
          shape match {
            case Struct(t1 :: t2 :: Nil) =>
              val (s1, s2) = (t1.size, t2.size)
              (
                m1.constructIn(t1, expr.slice(0, s1)),
                m2.constructIn(t2, expr.slice(s1, s1 + s2)))
            case _ => error("StructShape of size 2 is required")
          }
        }
        override def deconstructOut(out: (m1.Out, m2.Out)): (Layout, Seq[Tensor[_]]) = {
          val (shape1, tensor1) = m1.deconstructOut(out._1)
          val (shape2, tensor2) = m2.deconstructOut(out._2)
          (Struct(Seq(shape1, shape2)), tensor1 ++ tensor2)
        }
        override def constructOutRaw(shape: Layout, tensors: Seq[RawTensor]): Out = {
          shape match {
            case Struct(t1 :: t2 :: Nil) =>
              val (s1, s2) = (t1.size, t2.size)
              (
                m1.constructOutRaw(t1, tensors.slice(0, s1)),
                m2.constructOutRaw(t2, tensors.slice(s1, s1 + s2)))
            case _ => error("StructShape of size 2 is required")
          }
        }
      }

    implicit def matTuple3Expr[A1, A2, A3](implicit m1: Mat[A1], m2: Mat[A2], m3: Mat[A3]) =
      new Mat[(A1, A2, A3)] {
        override type Out = (m1.Out, m2.Out, m3.Out)
        override def deconstructIn(in: (A1, A2, A3)): (Layout, Seq[Expr[_]]) = {
          val (t1, expr1) = m1.deconstructIn(in._1)
          val (t2, expr2) = m2.deconstructIn(in._2)
          val (t3, expr3) = m3.deconstructIn(in._3)
          (Struct(Seq(t1, t2, t3)), expr1 ++ expr2 ++ expr3)
        }
        override def constructIn(shape: Layout, expr: Seq[Expr[_]]): (A1, A2, A3) = {
          shape match {
            case Struct(t1 :: t2 :: t3 :: Nil) =>
              val (s1, s2, s3) = (t1.size, t2.size, t3.size)
              (
                m1.constructIn(t1, expr.slice(0, s1)),
                m2.constructIn(t2, expr.slice(s1, s1 + s2)),
                m3.constructIn(t3, expr.slice(s1 + s2, s1 + s2 + s3)))
            case _ => error("StructShape of size 3 is required")
          }
        }
        override def deconstructOut(out: (m1.Out, m2.Out, m3.Out)): (Layout, Seq[Tensor[_]]) = {
          val (t1, tensor1) = m1.deconstructOut(out._1)
          val (t2, tensor2) = m2.deconstructOut(out._2)
          val (t3, tensor3) = m3.deconstructOut(out._3)
          (Struct(Seq(t1, t2, t3)), tensor1 ++ tensor2 ++ tensor3)
        }
        override def constructOutRaw(shape: Layout, tensors: Seq[RawTensor]): Out = {
          shape match {
            case Struct(t1 :: t2 :: t3 :: Nil) =>
              val (s1, s2, s3) = (t1.size, t2.size, t3.size)
              (
                m1.constructOutRaw(t1, tensors.slice(0, s1)),
                m2.constructOutRaw(t2, tensors.slice(s1, s1 + s2)),
                m3.constructOutRaw(t3, tensors.slice(s1 + s2, s1 + s2 + s3)))
            case _ => error("StructShape of size 3 is required")
          }
        }
      }
  }

  object syntax extends AllSyntax
}
