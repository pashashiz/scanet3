package org.scanet.core

import org.scanet.core.ConstOp.syntax._
import org.scanet.core.Output.Grad
import org.scanet.core.TensorType.syntax._
import org.scanet.core.Slice.syntax._
import org.scanet.math.{Floating, Numeric}
import simulacrum.{op, typeclass}

@typeclass trait CoreOp[F[_]] {

  /** Adds label to the output
   *
   * @param label to add
   * @return an output with a label attached
   */
  def as[A: TensorType](out: F[A], label: String): F[A]

  def slice[A: TensorType](out: F[A], projection: Projection): F[A]

  def slice[A: TensorType, S1: CanBuildSliceFrom](out: F[A], s1: S1): F[A] = slice(out, Projection(s1))
  def slice[A: TensorType, S1: CanBuildSliceFrom, S2: CanBuildSliceFrom](out: F[A], s1: S1, s2: S2): F[A] = slice(out, Projection(s1, s2))
  def slice[A: TensorType, S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom](out: F[A], s1: S1, s2: S2, s3: S3): F[A] = slice(out, Projection(s1, s2, s3))

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

  def joinAlong[A: TensorType](op: F[A], other: F[A], dim: Int): F[A]

  def join[A: TensorType](op: F[A], other: F[A]): F[A]

  def zip[A: TensorType](first: F[A], second: F[A]): F[A]

  def unzip[A: TensorType](zipped: F[A]): (F[A], F[A])
  def unzip3[A: TensorType](zipped: F[A]): (F[A], F[A], F[A])
  def unzipN[A: TensorType](zipped: F[A]): Seq[F[A]]

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

    implicit def coreOps: CoreOp[Output] = new CoreOp[Output] with CoreStandaloneOps {

      override def slice[A: TensorType](out: Output[A], projection: Projection): Output[A] = {
        val adjusted = projection.adjustTo(out.shape)
        val (offsets, lengths) = adjusted.asOffsetAndLength
        val sliced = Output.name[A]("Slice")
          .shape(adjusted.shapeFull)
          .inputs(
            out,
            as(Tensor.vector(offsets).const, "offsets"),
            as(Tensor.vector(lengths).const, "lengths"))
          .compileWithAllInputs
          .build
        // in case there is indexing and not really a slicing, like tensor.slice(1)
        // we would need to prune first dimensions
        if (adjusted.canPrune > 0) {
          reshape(sliced, adjusted.shapePruned)
        } else {
          sliced
        }
      }

      override def joinAlong[A: TensorType](op: Output[A], other: Output[A], dim: Int): Output[A] = joinAlong(Seq(op, other), dim)

      override def join[A: TensorType](op: Output[A], other: Output[A]): Output[A] = joinAlong(Seq(op, other), 0)

      override def cast[A: TensorType, B: TensorType](op: Output[A]): Output[B] = {
        if (TensorType[A] == TensorType[B]) op.asInstanceOf[Output[B]]
        else {
          Output.name[B]("Cast")
            .shape(op.shape)
            .inputs(op)
            .localGrad(new Grad[B] {
              override def calc[R: Numeric : Floating : TensorType](current: Output[B], parentGrad: Output[R]): List[Output[R]] = {
                List(parentGrad)
              }
            })
            .compileWithAttr("DstT", TensorType[B])
            .compileWithAllInputs
            .build
        }
      }

      override def asVoid[A: TensorType](op: Output[A]): Output[Nothing] =
        op.asInstanceOf[Output[Nothing]]

      override def zip[A: TensorType](first: Output[A], second: Output[A]): Output[A] = zip(Seq(first, second): _*)

      override def unzip[A: TensorType](zipped: Output[A]): (Output[A], Output[A]) = {
        require(zipped.shape.rank > 0, "cannot unzip a scalar")
        require(zipped.shape.head == 2, s"first dimension should be equal to 2 but was ${zipped.shape.head}")
        unzipN(zipped) match { case Seq(first, second) => (first, second) }
      }

      override def unzip3[A: TensorType](zipped: Output[A]): (Output[A], Output[A], Output[A]) = {
        require(zipped.shape.rank > 0, "cannot unzip a scalar")
        require(zipped.shape.head == 3, s"first dimension should be equal to 3 but was ${zipped.shape.head}")
        unzipN(zipped) match { case Seq(first, second, third) => (first, second, third) }
      }

      override def unzipN[A: TensorType](zipped: Output[A]): Seq[Output[A]] = {
        require(zipped.shape.rank > 0, "cannot unzip a scalar")
        val shape = zipped.shape.remove(0)
        (0 until zipped.shape.head).map(i => reshape(slice(zipped, i), shape))
      }
    }
  }

  trait CoreStandaloneOps {

    def as[A: TensorType](out: Output[A], label: String): Output[A] = out.copy(label = label)

    def placeholder[A: TensorType](shape: Shape): Output[A] = {
      Output.name[A]("Placeholder")
        .shape(shape)
        .compileWithAttr("dtype", TensorType[A])
        .compileWithAttr("shape", shape)
        .build
    }

    def placeholder[A: TensorType](shape: Int*): Output[A] = placeholder(Shape(shape: _*))

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

    def zip[A: TensorType](outputs: Output[A]*): Output[A] = {
      require(outputs.map(_.shape).distinct.size == 1, s"shapes should be equal but was ${outputs.map(_.shape)}")
      val shape = Shape(1 :: outputs.head.shape.dims)
      joinAlong(outputs.map(reshape(_, shape)), 0)
    }

    def joinAlong[A: TensorType](outputs: Seq[Output[A]], axis: Int): Output[A] = {
      val shapes = outputs.map(_.shape)
      require(shapes.map(_.rank).distinct.size == 1,
        s"all inputs should have same rank, but was ${shapes.mkString(", ")}")
      require(shapes.map(_.remove(axis)).distinct.size == 1,
        s"all inputs should have same dimensions except the axis, but was ${shapes.mkString(", ")}")
      val newDimSize = shapes.map(_.dims(axis)).sum
      val shape = Shape(shapes.head.dims.updated(axis, newDimSize))
      Output.name[A]("ConcatV2")
        .shape(shape)
        .inputs(outputs :+ as(Tensor.scalar(axis.toLong).const, "axis"): _*)
        .compileWithTransformer((ctx, builder) => {
          val compiledInputs = ctx.inputs.map(_.output(0))
          builder
            .addInputList(compiledInputs.take(outputs.size).toArray)
            .addInput(compiledInputs.last)
        })
        .build
    }

    def squeeze[A: TensorType](op: Output[A]): Output[A] = {
      val squeezed = op.shape.squeeze
      if (squeezed.rank < op.shape.rank) {
        Output.name[A]("Squeeze")
          .shape(squeezed)
          .inputs(op)
          .localGrad(new Grad[A] {
            override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
              List(reshape(parentGrad, op.shape))
            }
          })
          .compileWithAllInputs
          .build
      } else {
        op
      }
    }

    def reshape[A: TensorType](op: Output[A], shape: Shape): Output[A] = {
      require(op.shape.power == shape.power ,
        s"shape ${op.shape} cannot be reshaped into $shape")
      if (op.shape != shape) {
        // note: scalar is a special case, reshape does not work with scalars
        if (shape.isScalar) {
          squeeze(op)
        } else {
          Output.name[A]("Reshape")
            .shape(shape)
            .inputs(op, as(Tensor.vector(shape.dims: _*).const, "new_shape"))
            .localGrad(new Grad[A] {
              override def calc[R: Numeric : Floating : TensorType](current: Output[A], parentGrad: Output[R]): List[Output[R]] = {
                List(reshape(parentGrad, op.shape))
              }
            })
            .compileWithAllInputs
            .build
        }
      } else {
        op
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

  trait Syntax extends Instances with CoreOp.ToCoreOpOps with CoreStandaloneOps
  object syntax extends Syntax
}