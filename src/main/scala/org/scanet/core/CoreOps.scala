package org.scanet.core

import simulacrum.typeclass
import org.scanet.core.TfType.syntax._
import org.scanet.core.Const.syntax._

@typeclass trait CoreOps[A] {
  def as(op: A, label: String): A
  def reshape(op: A, shape: Shape): A
  // note: shapeless failed to use Int*
  def reshape(op: A, dim1: Int): A = reshape(op, Shape(dim1))
  def reshape(op: A, dim1: Int, dim2: Int): A = reshape(op, Shape(dim1, dim2))
  def reshape(op: A, dim1: Int, dim2: Int, dim3: Int): A = reshape(op, Shape(dim1, dim2, dim3))
  def squeeze(op: A): A
}

object CoreOps {

  trait Instances {

    implicit def coreOps[A: TfType]: CoreOps[Output[A]] = new CoreOps[Output[A]] {

      override def as(op: Output[A], label: String): Output[A] = op.copy(label = label)

      override def reshape(op: Output[A], shape: Shape): Output[A] = {
        require(op.shape.power == shape.power ,
          s"shape ${op.shape} cannot be reshaped into $shape")
        if (op.shape != shape) {
          // note: scalar is a special case, reshape does not work with scalars
          if (shape.isScalar) {
            squeeze(op)
          } else {
            Output.name[A]("Reshape")
              .shape(shape)
              .inputs(op, Tensor.vector(shape.dims: _*).const)
              .compileWithAllInputs
              .build
          }
        } else {
          op
        }
      }

      override def squeeze(op: Output[A]): Output[A] = {
        val squeezed = op.shape.squeeze
        if (squeezed.rank < op.shape.rank) {
          Output.name[A]("Squeeze")
            .shape(squeezed)
            .inputs(op)
            .compileWithAllInputs
            .build
        } else {
          op
        }
      }
    }
  }
  trait Syntax extends Instances with CoreOps.ToCoreOpsOps
  object syntax extends Syntax
}