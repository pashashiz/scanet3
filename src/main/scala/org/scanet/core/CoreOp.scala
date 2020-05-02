package org.scanet.core

import org.scanet.core.Const.syntax._
import org.scanet.core.TensorType.syntax._
import simulacrum.{op, typeclass}

import scala.language.higherKinds

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

  /** Asserts that given condition (constructed from current tensor by specified fn)
   * is true or fail graph execution (and prints current value).
   *
   * {{{
   * val a = Tensor.vector(1, 2).const
   * val b = Tensor.vector(3, 4).const
   * val c = Tensor.vector(4, 6).const
   *
   * (a plus b).assert(_ === c).eval should be(Tensor.vector(4, 6))
   * }}}
   *
   * @param f function to build assertion condition from current op
   * @return current op for chaining
   */
  def assert[A: TensorType](op: F[A], f: F[A] => F[Boolean]): F[A]

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
  def asLeaf[A: TensorType](op: F[A]): F[Nothing]
}

object CoreOp {

  trait Instances {

    implicit def coreOps: CoreOp[Output] = new CoreOp[Output] with OutputCoreOps {

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
        Output.name[A]("Identity")
          .shape(op.shape)
          .inputs(op)
          .controlInputs(dep)
          .compileWithAllInputs
          .compileWithControlInputs
          .build
      }

      override def assert[A: TensorType](op: Output[A], f: Output[A] => Output[Boolean]): Output[A] = {
        dependsOn(op, assertThat(f(op), op))
      }
    }
  }

  trait OutputCoreOps {

    /** Asserts that given condition is true or fail graph execution.
     * Optional outputs can be specified to print on assertion failure.
     *
     * {{{
     * val a = 2.const
     * val b = 1.const
     * val c = (a div b) << assertThat(a gt b)
     * c.eval should be(Tensor.scalar(2))
     * }}}
     *
     * @param cond  assertion condition
     * @param print outputs to print on assert error
     * @see dependsOn
     * @return leaf node to add as dependant op
     */
    def assertThat(cond: Output[Boolean], print: Output[_]*): Output[Nothing] = {
      val ops = if (print.isEmpty) List("assertion error".const) else print.toList
      val assert = Output.name[Boolean]("Assert")
        .shape(Shape())
        .inputs(cond :: ops: _*)
        .compileWithTransformer((ctx, builder) => {
          // add condition as input
          builder.addInput(ctx.inputs.head.output(0))
          // add ops to print as input list (has to be not empty list)
          builder.addInputList(ctx.inputs.tail.map(_.output(0)).toArray)
        })
        .build
      asLeaf(assert)
    }

    def asLeaf[A: TensorType](op: Output[A]): Output[Nothing] = {
      op.asInstanceOf[Output[Nothing]]
    }
  }

  trait Syntax extends Instances with CoreOp.ToCoreOpOps with OutputCoreOps
  object syntax extends Syntax
}