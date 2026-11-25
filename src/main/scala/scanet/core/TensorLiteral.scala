package scanet.core

import scala.collection.mutable.ArrayBuffer

/** Describes how a nested array literal is turned into a tensor, so it is easy to read
  * and write tensors of any rank in a way they are usually visualized:
  * {{{
  * Tensor(Array(
  *   Array(Array(1, 2), Array(3, 4)),
  *   Array(Array(5, 6), Array(7, 8))))
  * }}}
  * The rank of a tensor is equal to the depth of nested arrays,
  * all the arrays on the same level must have the same (non zero) length.
  *
  * Any [[scala.collection.Iterable]] (`Seq`, `List`, `Vector`...) can be used instead of an array,
  * and they can be mixed, like `Array(List(1, 2), List(3, 4))`.
  */
trait TensorLiteral[T, A] {
  def shape(literal: T): Shape
  def flatten(literal: T, into: ArrayBuffer[A]): Unit
}

object TensorLiteral {

  implicit def elements[A: TensorType]: TensorLiteral[Array[A], A] =
    new TensorLiteral[Array[A], A] {
      override def shape(literal: Array[A]): Shape = {
        require(literal.nonEmpty, "tensor literal cannot contain empty arrays")
        Shape(literal.length)
      }
      override def flatten(literal: Array[A], into: ArrayBuffer[A]): Unit =
        into ++= literal
    }

  implicit def nested[T, A](implicit inner: TensorLiteral[T, A]): TensorLiteral[Array[T], A] =
    new TensorLiteral[Array[T], A] {
      override def shape(literal: Array[T]): Shape = {
        require(literal.nonEmpty, "tensor literal cannot contain empty arrays")
        val shapes = literal.toList.map(inner.shape).distinct
        require(
          shapes.size == 1,
          "all arrays on the same level of tensor literal must have the same shape, " +
          s"but got ${shapes.mkString(", ")}")
        Shape(literal.length :: shapes.head.dims)
      }
      override def flatten(literal: Array[T], into: ArrayBuffer[A]): Unit =
        literal.foreach(inner.flatten(_, into))
    }

  implicit def iterableElements[F[x] <: Iterable[x], A: TensorType]: TensorLiteral[F[A], A] =
    new TensorLiteral[F[A], A] {
      override def shape(literal: F[A]): Shape = {
        require(literal.nonEmpty, "tensor literal cannot contain empty arrays")
        Shape(literal.size)
      }
      override def flatten(literal: F[A], into: ArrayBuffer[A]): Unit =
        into ++= literal
    }

  implicit def iterableNested[F[x] <: Iterable[x], T, A](
      implicit inner: TensorLiteral[T, A]): TensorLiteral[F[T], A] =
    new TensorLiteral[F[T], A] {
      override def shape(literal: F[T]): Shape = {
        require(literal.nonEmpty, "tensor literal cannot contain empty arrays")
        val shapes = literal.toList.map(inner.shape).distinct
        require(
          shapes.size == 1,
          "all arrays on the same level of tensor literal must have the same shape, " +
          s"but got ${shapes.mkString(", ")}")
        Shape(literal.size :: shapes.head.dims)
      }
      override def flatten(literal: F[T], into: ArrayBuffer[A]): Unit =
        literal.foreach(inner.flatten(_, into))
    }
}
