package org.scanet.core

import simulacrum.typeclass
import org.scanet.core.TensorType.syntax._
import org.scanet.core.Const.syntax._

@typeclass trait CoreOp[A] {

  /** Adds label to the output
   *
   * @param label to add
   * @return an output with a label attached
   */
  def as(out: A, label: String): A

  /** Reshapes an output tensor.
    *
    * Requirements:
    * - the number of elements in old and new shape should be the same
    *
    * Example:
    * {{{
    * val a = Tensor.matrix(
    *   Array(1, 2, 3),
    *   Array(4, 5, 6))
    * val b = Tensor.matrix(
    *   Array(1, 2),
    *   Array(3, 4),
    *   Array(5, 6))
    * a.const.reshape(3, 2).eval should be(b)
    * }}}
    * @param shape a new shape
    * @return an output with new shape
    */
  def reshape(op: A, shape: Shape): A
  def reshape(op: A, dim1: Int): A = reshape(op, Shape(dim1))
  def reshape(op: A, dim1: Int, dim2: Int): A = reshape(op, Shape(dim1, dim2))
  def reshape(op: A, dim1: Int, dim2: Int, dim3: Int): A = reshape(op, Shape(dim1, dim2, dim3))

  /** Removes dimensions of size 1 from the shape of a tensor.
    *
    * Given a tensor, this operation returns a tensor of the same type with all dimensions of size 1 removed.
    *
    * @return squeezed output
    */
  def squeeze(op: A): A
}

object CoreOp {

  trait Instances {

    implicit def coreOps[A: TensorType]: CoreOp[Output[A]] = new CoreOp[Output[A]] {

      override def as(out: Output[A], label: String): Output[A] = out.copy(label = label)

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
  trait Syntax extends Instances with CoreOp.ToCoreOpOps
  object syntax extends Syntax
}