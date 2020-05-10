package org.scanet.core

import org.scanet.core.ConstOp.syntax._
import org.scanet.math.Numeric.syntax._
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

    implicit def coreOps: CoreOp[Output] = new CoreOp[Output] with ControlFlowOps {

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
              .localGrad(ctx => List(reshape(ctx.parentGrad, op.shape)))
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
            .localGrad(ctx => List(reshape(ctx.parentGrad, op.shape)))
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
            .localGrad(ctx => List(ctx.parentGrad))
            .compileWithAttr("DstT", TensorType[B])
            .compileWithAllInputs
            .build
        }
      }

      override def asVoid[A: TensorType](op: Output[A]): Output[Nothing] =
        op.asInstanceOf[Output[Nothing]]
    }
  }

  trait ControlFlowOps {

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
    def dependsOn[A: TensorType](op: Output[A], dep: Output[_]): Output[A] = {
      Output.name[A]("Identity")
        .shape(op.shape)
        .inputs(op)
        .controlInputs(dep)
        .compileWithAllInputs
        .compileWithControlInputs
        .build
    }

    /** Start build conditional operator by specifying boolean condition based on which
     * one of the specified outputs will be returned.
     *
     * Can be used to build ternary operators.
     *
     * {{{
     * val a = 1.const
     * val b = 0.const
     * val c = 2.const
     *
     * val ternary = when(a gt b) thenDo (a plus c) elseDo (a minus c)
     * ternary.eval should be(Tensor.scalar(3))
     * }}}
     *
     * @param cond if condition
     * @return current output
     */
    def when(cond: Output[Boolean]): ThenStep = new ThenStep {
      override def thenDo[A: TensorType](trueCase: Output[A]): ElseStep[A] = (falseCase: Output[A]) => {
        require(falseCase.shape == trueCase.shape)

        // perform switch op that sends input to 0 output when condition is false and to 1 for true
        val switch = Output.name[A]("Switch")
          .shape(Shape())
          .inputs(cond, cond)
          .compileWithAllInputs
          .build

        // outputs are first wrapped into Identity ops to select input with proper index
        // TODO: this identity ops can be removed if we can set input index to prebuilt Output
        val falseBranch = Output.name[A]("Identity")
          .inputs(switch)
          .shape(Shape())
          .compileWithAllInputsAtIndex(0)
          .build
        val trueBranch = Output.name[A]("Identity")
          .inputs(switch)
          .shape(Shape())
          .compileWithAllInputsAtIndex(1)
          .build

        // merge branches into single output (it selects first available input)
        Output.name[A]("Merge")
          .shape(trueCase.shape)
          .inputs(dependsOn(trueCase, trueBranch), dependsOn(falseCase, falseBranch))
          .compileWithInputList
          .build
      }
    }
  }

  trait ThenStep {

    /** Continue building conditional operator by specifying `true` branch output
     *
     * @param op Output to return when specified condition is true
     * @tparam A type of conditional expression
     * @return else branch specification
     */
    def thenDo[A: TensorType](op: Output[A]): ElseStep[A]
  }

  trait ElseStep[A] {

    /** Finish building conditional operator by specifying `true` branch output
     *
     * @param op Output to return when specified condition is false
     * @return conditional (ternary) operation
     */
    def elseDo(op: Output[A]): Output[A]
  }

  trait Syntax extends Instances with CoreOp.ToCoreOpOps with ControlFlowOps
  object syntax extends Syntax
}