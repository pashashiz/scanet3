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
  // given MyType[Expr] deconstruct as Layout + Seq[Expr] so we could feed a session
  def deconstructIn(in: In): (Layout, Seq[Expr[_]])
  // given placeholders and layout, construct In representation so we could to pass into the session
  def constructIn(layout: Layout, expr: Seq[Expr[_]]): In
  // given MyType[Tensor] deconstruct as Layout + Seq[Tensor] so we could create placeholders
  def deconstructOut(out: Out): (Layout, Seq[Tensor[_]])
  // given raw tensors and layout, construct MyType[Tensor], that is done to get results after session run
  def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Out
}

sealed trait Layout {
  def size: Int
}

object Layout {
  case object Value extends Layout {
    override def size: Int = 1
  }
  case class SeqLayout(items: Seq[Layout]) extends Layout {
    override def size: Int = items.map(_.size).sum
  }
  case class MapLayout(items: Seq[(Any, Layout)]) extends Layout {
    override def size: Int = items.map(_._2.size).sum
  }
  case class Struct(children: Seq[Layout]) extends Layout {
    def apply(index: Int): Layout = children(index)
    def size: Int = children.map(_.size).sum
  }
}

class ValueMat[A: TensorType] extends Mat[Expr[A]] {
  override type Out = Tensor[A]
  override def deconstructIn(in: Expr[A]): (Layout, Seq[Expr[_]]) = (Value, Seq(in))
  override def constructIn(layout: Layout, expr: Seq[Expr[_]]): Expr[A] = layout match {
    case Value => expr.head.asInstanceOf[Expr[A]]
    case other => error(s"Unsupported layout $other")
  }
  override def deconstructOut(out: Tensor[A]): (Layout, Seq[Tensor[_]]) = (Value, Seq(out))
  override def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Out =
    layout match {
      case Value => Tensor.wrap[A](tensors.head)
      case other => error(s"Unsupported layout $other")
    }
}

object Mat {

  trait AllSyntax {

    implicit def valueMat[A: TensorType]: ValueMat[A] = new ValueMat[A]
    // Note: we have to define anonymous class so dependant types would work
    // that is probably caused by the fact that we have to preserve an exact structural type

    implicit def seqMat[A](implicit m: Mat[A]) = new Mat[Seq[A]] {
      override type Out = Seq[m.Out]
      override def deconstructIn(in: Seq[A]): (Layout, Seq[Expr[_]]) = {
        val (layouts, allExpr) = in.map(m.deconstructIn).unzip
        (SeqLayout(layouts), allExpr.flatten)
      }
      override def constructIn(layout: Layout, expr: Seq[Expr[_]]): Seq[A] = {
        layout match {
          case SeqLayout(items) =>
            items.foldLeft((Seq.empty[A], expr)) {
              case ((result, exprAll), next) =>
                val (consumed, rest) = exprAll.splitAt(next.size)
                val constructed = m.constructIn(next, consumed)
                (constructed +: result, rest)
            }._1.reverse
          case other => error(s"Unsupported layout $other")
        }
      }
      override def deconstructOut(out: Seq[m.Out]): (Layout, Seq[Tensor[_]]) = {
        val (layouts, allTensors) = out.map(m.deconstructOut).unzip
        (SeqLayout(layouts), allTensors.flatten)
      }
      override def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Seq[m.Out] = {
        layout match {
          case SeqLayout(items) =>
            items.foldLeft((Seq.empty[m.Out], tensors)) {
              case ((result, tensorsAll), next) =>
                val (consumed, rest) = tensorsAll.splitAt(next.size)
                val constructed = m.constructOutRaw(next, consumed)
                (constructed +: result, rest)
            }._1.reverse
          case other => error(s"Unsupported layout $other")
        }
      }
    }

    implicit def mapMat[K, V](implicit m: Mat[V]) = new Mat[Map[K, V]] {
      override type Out = Map[K, m.Out]
      override def deconstructIn(in: Map[K, V]): (Layout, Seq[Expr[_]]) = {
        val (layouts, allExpr) = in
          .map {
            case (key, value) =>
              val (layout, expr) = m.deconstructIn(value)
              ((key, layout), expr)
          }
          .toList
          .unzip
        (MapLayout(layouts), allExpr.flatten)
      }
      override def constructIn(layout: Layout, expr: Seq[Expr[_]]): Map[K, V] = {
        layout match {
          case MapLayout(items) =>
            items.foldLeft((Map.empty[K, V], expr)) {
              case ((result, exprAll), (key, next)) =>
                val (consumed, rest) = exprAll.splitAt(next.size)
                val constructed = m.constructIn(next, consumed)
                (result + (key.asInstanceOf[K] -> constructed), rest)
            }._1
          case other => error(s"Unsupported layout $other")
        }
      }
      override def deconstructOut(out: Map[K, m.Out]): (Layout, Seq[Tensor[_]]) = {
        val (layouts, allTensors) = out
          .map {
            case (key, value) =>
              val (layout, tensors) = m.deconstructOut(value)
              ((key, layout), tensors)
          }
          .toList
          .unzip
        (MapLayout(layouts), allTensors.flatten)
      }
      override def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Out = {
        layout match {
          case MapLayout(items) =>
            items.foldLeft((Map.empty[K, m.Out], tensors)) {
              case ((result, tensorsAll), (key, next)) =>
                val (consumed, rest) = tensorsAll.splitAt(next.size)
                val constructed = m.constructOutRaw(next, consumed)
                (result + (key.asInstanceOf[K] -> constructed), rest)
            }._1
          case other => error(s"Unsupported layout $other")
        }
      }
    }

    // check to see if we can implement xmap with dependant types
    implicit def paramsMat[A](implicit m: Mat[A]) = new Mat[Params[A]] {
      override type Out = Params[m.Out]
      private val map = mapMat[Path, A]
      override def deconstructIn(in: Params[A]): (Layout, Seq[Expr[_]]) =
        map.deconstructIn(in.params)
      override def constructIn(layout: Layout, expr: Seq[Expr[_]]): Params[A] =
        Params(map.constructIn(layout, expr))
      override def deconstructOut(out: Params[m.Out]): (Layout, Seq[Tensor[_]]) =
        map.deconstructOut(out.params)
      override def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Out =
        Params(map.constructOutRaw(layout, tensors))
    }

    implicit def tuple2Mat[A1, A2](implicit m1: Mat[A1], m2: Mat[A2]) =
      new Mat[(A1, A2)] {
        override type Out = (m1.Out, m2.Out)
        override def deconstructIn(in: (A1, A2)): (Layout, Seq[Expr[_]]) = {
          val (shape1, expr1) = m1.deconstructIn(in._1)
          val (shape2, expr2) = m2.deconstructIn(in._2)
          (Struct(Seq(shape1, shape2)), expr1 ++ expr2)
        }
        override def constructIn(layout: Layout, expr: Seq[Expr[_]]): (A1, A2) = {
          layout match {
            case Struct(t1 :: t2 :: Nil) =>
              val (s1, s2) = (t1.size, t2.size)
              (
                m1.constructIn(t1, expr.slice(0, s1)),
                m2.constructIn(t2, expr.slice(s1, s1 + s2)))
            case _ => error("Struct layout of size 2 is required")
          }
        }
        override def deconstructOut(out: (m1.Out, m2.Out)): (Layout, Seq[Tensor[_]]) = {
          val (shape1, tensor1) = m1.deconstructOut(out._1)
          val (shape2, tensor2) = m2.deconstructOut(out._2)
          (Struct(Seq(shape1, shape2)), tensor1 ++ tensor2)
        }
        override def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Out = {
          layout match {
            case Struct(t1 :: t2 :: Nil) =>
              val (s1, s2) = (t1.size, t2.size)
              (
                m1.constructOutRaw(t1, tensors.slice(0, s1)),
                m2.constructOutRaw(t2, tensors.slice(s1, s1 + s2)))
            case _ => error("StructShape of size 2 is required")
          }
        }
      }

    implicit def tuple3Mat[A1, A2, A3](implicit m1: Mat[A1], m2: Mat[A2], m3: Mat[A3]) =
      new Mat[(A1, A2, A3)] {
        override type Out = (m1.Out, m2.Out, m3.Out)
        override def deconstructIn(in: (A1, A2, A3)): (Layout, Seq[Expr[_]]) = {
          val (t1, expr1) = m1.deconstructIn(in._1)
          val (t2, expr2) = m2.deconstructIn(in._2)
          val (t3, expr3) = m3.deconstructIn(in._3)
          (Struct(Seq(t1, t2, t3)), expr1 ++ expr2 ++ expr3)
        }
        override def constructIn(layout: Layout, expr: Seq[Expr[_]]): (A1, A2, A3) = {
          layout match {
            case Struct(t1 :: t2 :: t3 :: Nil) =>
              val (s1, s2, s3) = (t1.size, t2.size, t3.size)
              (
                m1.constructIn(t1, expr.slice(0, s1)),
                m2.constructIn(t2, expr.slice(s1, s1 + s2)),
                m3.constructIn(t3, expr.slice(s1 + s2, s1 + s2 + s3)))
            case _ => error("Struct layout of size 3 is required")
          }
        }
        override def deconstructOut(out: (m1.Out, m2.Out, m3.Out)): (Layout, Seq[Tensor[_]]) = {
          val (t1, tensor1) = m1.deconstructOut(out._1)
          val (t2, tensor2) = m2.deconstructOut(out._2)
          val (t3, tensor3) = m3.deconstructOut(out._3)
          (Struct(Seq(t1, t2, t3)), tensor1 ++ tensor2 ++ tensor3)
        }
        override def constructOutRaw(layout: Layout, tensors: Seq[RawTensor]): Out = {
          layout match {
            case Struct(t1 :: t2 :: t3 :: Nil) =>
              val (s1, s2, s3) = (t1.size, t2.size, t3.size)
              (
                m1.constructOutRaw(t1, tensors.slice(0, s1)),
                m2.constructOutRaw(t2, tensors.slice(s1, s1 + s2)),
                m3.constructOutRaw(t3, tensors.slice(s1 + s2, s1 + s2 + s3)))
            case _ => error("Struct layout of size 3 is required")
          }
        }
      }
  }

  object syntax extends AllSyntax
}
