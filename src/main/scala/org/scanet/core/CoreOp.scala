package org.scanet.core

import org.scanet.core.Const.syntax._
import org.scanet.core.TensorType.syntax._
import simulacrum.{op, typeclass}

@typeclass trait CoreOp[F[_]] {

  /** Adds label to the output
   *
   * @param label to add
   * @return an output with a label attached
   */
  def as[A: TensorType](out: F[A], label: String): F[A]

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
  def reshape[A: TensorType](op: F[A], shape: Shape): F[A]
  def reshape[A: TensorType](op: F[A], dim1: Int): F[A] = reshape(op, Shape(dim1))
  def reshape[A: TensorType](op: F[A], dim1: Int, dim2: Int): F[A] = reshape(op, Shape(dim1, dim2))
  def reshape[A: TensorType](op: F[A], dim1: Int, dim2: Int, dim3: Int): F[A] = reshape(op, Shape(dim1, dim2, dim3))

  /** Removes dimensions of size 1 from the shape of a tensor.
    *
    * Given a tensor, this operation returns a tensor of the same type with all dimensions of size 1 removed.
    *
    * @return squeezed output
    */
  def squeeze[A: TensorType](op: F[A]): F[A]

  /** Add operation which will be executed right after current operation and
   * return current operation as output to continue chaining.
   *
   * Can be used to add logging or assert operations.
   *
   * {{{
   * val a = 1.const
   * val b = 2.const
   * val c = (a plus b) << print("a + b = {} + {}", a, b)
   * c.eval should be(Tensor.scalar(3)) // and prints `a + b = 1 + 2` before performing plus op
   * }}}
   *
   * @param dep dependant leaf operation
   * @return current output
   */
  @op("<<", alias = true)
  def dependsOn[A: TensorType](op: F[A], dep: F[_]): F[A]

  /** If operation - performs evaluates `trueCase` branch when given condition
   * is `true` and `falseCase` otherwise
   *
   * {{{
   * val a = 1.const
   * val b = 0.const
   * val c = 2.const
   *
   * a.when(_ gt b, _ plus c, _ minus c).eval should be(Tensor.scalar(3))
   * }}}
   *
   * @param cond predicate
   * @param trueCase function to build true branch
   * @param falseCase function to build false branch
   * @return output based on given condition
   */
  def when[A: TensorType, B: TensorType](
                                          op: F[A],
                                          cond: F[A] => F[Boolean],
                                          trueCase: F[A] => F[B],
                                          falseCase: F[A] => F[B]
                                        ): F[B]

  /** Cast elements of given tensor form type A into B.
   * Returns given input if A is already equal to B.
   *
   * @return casted output
   */
  def cast[A: TensorType, B: TensorType](op: F[A]): F[B]

  /** Return current op as leaf operation - means it doesnt produce output
   * and can be evaluated in graph only if added as dependant operation.
   *
   * @see dependsOn
   * @return current leaf node
   */
  def asVoid[A: TensorType](op: F[A]): F[Nothing]
}

object CoreOp {

  trait Instances {

    implicit def coreOps: CoreOp[Output] = new CoreOp[Output] {

      override def as[A: TensorType](out: Output[A], label: String): Output[A] = out.copy(label = label)

      override def reshape[A: TensorType](op: Output[A], shape: Shape): Output[A] = {
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

      override def squeeze[A: TensorType](op: Output[A]): Output[A] = {
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

      override def cast[A: TensorType, B: TensorType](op: Output[A]): Output[B] = {
        if (TensorType[A] == TensorType[B]) op.asInstanceOf[Output[B]]
        else {
          Output.name[B]("Cast")
            .shape(op.shape)
            .inputs(op)
            .compileWithAttr("DstT", TensorType[B])
            .compileWithAllInputs
            .build
        }
      }

      override def dependsOn[A: TensorType](op: Output[A], dep: Output[_]): Output[A] = {
        identity(op)
          .controlInputs(dep)
          .compileWithAllInputs
          .compileWithControlInputs
          .build
      }

      override def when[A: TensorType, B: TensorType](
                                                       op: Output[A],
                                                       cond: Output[A] => Output[Boolean],
                                                       trueCase: Output[A] => Output[B],
                                                       falseCase: Output[A] => Output[B]
                                                     ): Output[B] = {
        // perform switch op that sends input to 0 output when condition is false and to 1 for true
        val switch = Output.name[A]("Switch")
          .shape(op.shape)
          .inputs(op, cond(op))
          .compileWithAllInputs
          .build

        // outputs are first wrapped into Identity ops to select input with proper index
        // TODO: this identity ops can be removed if we can set input index to prebuilt Output
        val falseBranch = identity(switch)
          .compileWithAllInputsAtIndex(0)
          .build
        val trueBranch = identity(switch)
          .compileWithAllInputsAtIndex(1)
          .build

        // merge branches into single output (it selects first available output)
        Output.name[B]("Merge")
          .shape(op.shape)
          .inputs(trueCase(trueBranch), falseCase(falseBranch))
          .compileWithInputList
          .build
      }

      override def asVoid[A: TensorType](op: Output[A]): Output[Nothing] =
        op.asInstanceOf[Output[Nothing]]

      private def identity[A: TensorType](op: Output[A]) =
        Output.name[A]("Identity")
          .shape(op.shape)
          .inputs(op)
    }
  }
  trait Syntax extends Instances with CoreOp.ToCoreOpOps
  object syntax extends Syntax
}