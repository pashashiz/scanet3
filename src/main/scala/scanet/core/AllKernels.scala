package scanet.core

import org.tensorflow.OperationBuilder
import scanet.core.Const.syntax._
import scanet.core.DefaultCompiler.Ctx
import scanet.core.Slice.syntax._
import scanet.core.TensorType.syntax._

import scala.collection.immutable.Seq

case class Placeholder[A: TensorType](shape: Shape) extends Expr[A] {
  override def name: String = "Placeholder"
  override val tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def id: Option[String] = Some(s"#$address")
  override def inputs: Seq[Expr[_]] = Seq.empty
  override def compiler: Compiler[A] =
    DefaultCompiler[A]().withAttr("dtype", TensorType[A]).withAttr("shape", shape)
}

/** Creates a tensor filled with a scalar value.
  *
  * For example:
  * {{{
  * fill(2, 2)(1).eval should be(Tensor.matrix(Array(1, 1), Array(1, 1)))
  * }}}
  *
  * `fill` differs from `const` in a few ways:
  *  - `fill` only supports scalar contents, whereas `const` supports Tensor values
  *  - `fill` creates an Op in the computation graph that constructs the actual
  *     Tensor value at runtime. This is in contrast to `const` which embeds
  *     the entire Tensor into the graph with a `Const` node.
  *  - `fill` is a subject of caching and sub-graph reusing even with high rank cause
  *     that is possible to check value equality when in contract to a large `const` tensor event when
  *     filled with same value it is impossible to compare with other existing constants by value
  *     during graph construction
  *
  * @param shape a shape to fill the tensor with
  * @param scalar a value
  * @return output
  */
case class Fill[A: TensorType] private (shape: Shape, scalar: Expr[A]) extends Expr[A] {
  require(scalar.isScalar, s"value should be a scalar but has shape ${scalar.shape}")
  override def name: String = "Fill"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val inputs: Seq[Expr[_]] = Seq(Tensor.vector(shape.dims: _*).const.as("shape"), scalar)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    import scanet.math.alg.kernels._
    override def calc[R:Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = Seq(null, sum(parentGrad))
  }
}

object Fill {
  def apply[A: TensorType](shape: Shape, scalar: Expr[A]): Expr[A] = {
    require(scalar.isScalar, s"value should be a scalar but has shape ${scalar.shape}")
    if (shape.isScalar) {
      scalar
    } else {
      new Fill[A](shape, scalar)
    }
  }
}

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
case class DependsOn[A: TensorType](expr: Expr[A], dep: Expr[_]) extends Expr[A] {
  override def name: String = "Identity"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def controls: Seq[Expr[_]] = Seq(dep)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
}

case class Switch[A: TensorType](cond: Expr[Boolean], output: Expr[A]) extends Expr[(A, A)] {
  override def name: String = "Switch"
  override def tpe: Option[TensorType[(A, A)]] = None
  override def shape: Shape = output.shape
  override def inputs: Seq[Expr[_]] = Seq(output, cond)
  override def compiler: Compiler[(A, A)] = DefaultCompiler[(A, A)](index = None)
}

trait TakeOutExpr[A] extends Expr[A] {
  def from: Expr[_]
  def index: Int
  def shape: Shape
  override def name: String = "Identity"
  override def inputs: Seq[Expr[_]] = Seq(from)
  override def compiler: Compiler[A] = DefaultCompiler[A](withInputs = false).withStage {
    (ctx: Ctx, builder: OperationBuilder) =>
      builder.addInput(ctx.inputs.head.operation.output(index))
  }
  override def toStringChild: String = s"($from[$index])"
}

case class TakeOut[A: TensorType](from: Expr[_], index: Int, shape: Shape) extends TakeOutExpr[A] {
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
}

case class TakeOutUntyped[A](from: Expr[_], index: Int, shape: Shape) extends TakeOutExpr[A] {
  override def tpe: Option[TensorType[A]] = None
}

case class Merge[A: TensorType](expr: Seq[Expr[A]]) extends Expr[A] {
  require(expr.size >= 2, "there should be at least 2 expressions")
  require(expr.map(_.shape).toSet.size == 1, "all expressions should have the same shape")
  override def name: String = "Merge"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.head.shape
  override def inputs: Seq[Expr[_]] = expr
  override def compiler: Compiler[A] = DefaultCompiler[A](inputsAsList = true)
}

case class Squeeze[A: TensorType] private (expr: Expr[A]) extends Expr[A] {
  override def name: String = "Squeeze"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = expr.shape.squeeze
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R:Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] =
      List(Reshape(parentGrad, expr.shape))
  }
}

object Squeeze {
  def apply[A: TensorType](expr: Expr[A]): Expr[A] = {
    val squeezed = expr.shape.squeeze
    if (squeezed.rank < expr.shape.rank) {
      new Squeeze[A](expr)
    } else {
      expr
    }
  }
}

case class Reshape[A: TensorType] private (expr: Expr[A], shape: Shape) extends Expr[A] {
  override def name: String = "Reshape"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val inputs: Seq[Expr[_]] = Seq(expr, Tensor.vector(shape.dims: _*).const.as("new_shape"))
  override def compiler: Compiler[A] = DefaultCompiler[A]()
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R:Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] =
      List(Reshape(parentGrad, expr.shape))
  }
}

object Reshape {
  def apply[A: TensorType](expr: Expr[A], shape: Shape): Expr[A] = {
    require(expr.shape.power == shape.power, s"shape ${expr.shape} cannot be reshaped into $shape")
    if (expr.shape != shape) {
      // note: scalar is a special case, reshape does not work with scalars
      if (shape.isScalar) {
        Squeeze(expr)
      } else {
        new Reshape(expr, shape)
      }
    } else {
      expr
    }
  }
}

case class SliceOp[A: TensorType] private (expr: Expr[A], projection: Projection) extends Expr[A] {
  private val (offsetsExpr, lengthsExpr) = {
    val (offsets, lengths) = projection.asOffsetAndLength
    (Tensor.vector(offsets).const.as("offsets"), Tensor.vector(lengths).const.as("lengths"))
  }
  override def name: String = "Slice"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = projection.shapeFull
  override def inputs: Seq[Expr[_]] = Seq(expr, offsetsExpr, lengthsExpr)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
}

object SliceOp {
  def apply[A: TensorType](expr: Expr[A], projection: Projection): Expr[A] = {
    val adjusted = projection.adjustTo(expr.shape)
    val sliced = new SliceOp[A](expr, adjusted)
    if (adjusted.canPrune > 0) {
      Reshape(sliced, adjusted.shapePruned)
    } else {
      sliced
    }
  }
}

case class JoinAlong[A: TensorType](outputs: Seq[Expr[A]], axis: Int) extends Expr[A] {
  import org.tensorflow.OperationBuilder
  import scanet.core.DefaultCompiler.Ctx

  private val shapes = outputs.map(_.shape)
  require(
    shapes.map(_.rank).distinct.size == 1,
    s"all inputs should have same rank, but was ${shapes.mkString(", ")}")
  require(
    shapes.map(_.remove(axis)).distinct.size == 1,
    s"all inputs should have same dimensions except the axis, but was ${shapes.mkString(", ")}")
  private val newDimSize = shapes.map(_.dims(axis)).sum
  override def name: String = "ConcatV2"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = Shape(shapes.head.dims.updated(axis, newDimSize))
  override val inputs: Seq[Expr[_]] = outputs :+ Tensor.scalar(axis.toLong).const.as("axis")
  override def compiler: Compiler[A] = DefaultCompiler[A](withInputs = false).withStage {
    (ctx: Ctx, builder: OperationBuilder) =>
      builder.addInputList(ctx.inputs.take(outputs.size).map(_.outputOrFail).toArray)
      builder.addInput(ctx.inputs.last.outputOrFail)
  }
  override def localGrad: Grad[A] = new Grad[A] {
    override def calc[R:Floating](
        current: Expr[A],
        parentGrad: Expr[R]): Seq[Expr[R]] = {
      val ranges = outputs
        .map(output => output.shape.dims(axis))
        .scanLeft(0 until 0)((prevRange, dimSize) => {
          prevRange.end until (prevRange.end + dimSize)
        })
        .tail
      val unbounded = parentGrad.shape.dims.map(dimSize => Slice(0, dimSize, isRange = true))
      ranges
        .map(range => {
          val slices = unbounded.updated(axis, Slice(range.start, range.end, isRange = true))
          SliceOp(parentGrad, Projection(slices))
        })
        .toList
    }
  }
}

case class Cast[B: TensorType] private (expr: Expr[_]) extends Expr[B] {
  override def name: String = "Cast"
  override def tpe: Option[TensorType[B]] = Some(TensorType[B])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[B] = DefaultCompiler[B]().withAttr("DstT", TensorType[B])
  override def localGrad: Grad[B] = new Grad[B] {
    override def calc[R:Floating](
        current: Expr[B],
        parentGrad: Expr[R]): Seq[Expr[R]] =
      List(parentGrad)
  }
}

object Cast {

  def safe[A: TensorType, B: TensorType](expr: Expr[A]): Expr[B] = {
    if (TensorType[A].tag == TensorType[B].tag)
      expr.asInstanceOf[Expr[B]]
    else
      new Cast[B](expr)
  }

  def unsafe[B: TensorType](expr: Expr[_]): Expr[B] = new Cast[B](expr)
}

trait AllKernels {

  def placeholder[A: TensorType](shape: Int*): Expr[A] = placeholder(Shape(shape: _*))
  def placeholder[A: TensorType](shape: Shape): Expr[A] = Placeholder[A](shape)

  def fill[A: TensorType](shape: Int*)(value: A): Expr[A] =
    fillOutput(Shape(shape: _*))(value.const)
  def fillOutput[A: TensorType](shape: Int*)(value: Expr[A]): Expr[A] =
    fillOutput(Shape(shape: _*))(value)
  def fill[A: TensorType](shape: Shape)(value: A): Expr[A] = fillOutput(shape)(value.const)
  def fillOutput[A: TensorType](shape: Shape)(value: Expr[A]): Expr[A] = Fill(shape, value)

  def dependsOn[A: TensorType](expr: Expr[A], dep: Expr[_]): Expr[A] = DependsOn(expr, dep)

  trait ThenStep {

    /** Continue building conditional operator by specifying `true` branch output
      *
      * @param op Output to return when specified condition is true
      * @tparam A type of conditional expression
      * @return else branch specification
      */
    def thenDo[A: TensorType](op: Expr[A]): ElseStep[A]
  }

  trait ElseStep[A] {

    /** Finish building conditional operator by specifying `true` branch output
      *
      * @param op Output to return when specified condition is false
      * @return conditional (ternary) operation
      */
    def elseDo(op: Expr[A]): Expr[A]
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
  def when(cond: Expr[Boolean]): ThenStep = new ThenStep {
    override def thenDo[A: TensorType](trueCase: Expr[A]): ElseStep[A] = (falseCase: Expr[A]) => {
      require(falseCase.shape == trueCase.shape, "shapes for true and false branches should match")
      // perform switch op that sends input to 0 output when condition is false and to 1 for true
      val switch = Switch(cond, cond)
      val falseBranch = TakeOut[Boolean](switch, 0, switch.shape)
      val trueBranch = TakeOut[Boolean](switch, 1, switch.shape)
      Merge(Seq(dependsOn(trueCase, trueBranch), dependsOn(falseCase, falseBranch)))
    }
  }

  def squeeze[A: TensorType](expr: Expr[A]): Expr[A] = Squeeze(expr)

  def reshape[A: TensorType](expr: Expr[A], shape: Shape): Expr[A] = Reshape(expr, shape)

  def slice[A: TensorType](expr: Expr[A], projection: Projection): Expr[A] =
    SliceOp(expr, projection)

  def joinAlong[A: TensorType](outputs: Seq[Expr[A]], axis: Int): Expr[A] = JoinAlong(outputs, axis)

  def zip[A: TensorType](outputs: Expr[A]*): Expr[A] = {
    require(
      outputs.map(_.shape).distinct.size == 1,
      s"shapes should be equal but was ${outputs.map(_.shape)}")
    val shape = Shape(1 :: outputs.head.shape.dims)
    joinAlong(outputs.toList.map(reshape(_, shape)), 0)
  }

  def cast[A: TensorType, B: TensorType](expr: Expr[A]): Expr[B] = Cast.safe(expr)
  def castUnsafe[B: TensorType](expr: Expr[_]): Expr[B] = Cast.unsafe[B](expr)
}

object kernels extends AllKernels {

  class TensorTypeOps[A: TensorType](expr: Expr[A]) {
    import scanet.core.{kernels => f}

    def slice(projection: Projection): Expr[A] = f.slice(expr, projection)
    def slice[S1: CanBuildSliceFrom](s1: S1): Expr[A] = slice(Projection(s1))
    def slice[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom](s1: S1, s2: S2): Expr[A] = slice(
      Projection(s1, s2))
    def slice[S1: CanBuildSliceFrom, S2: CanBuildSliceFrom, S3: CanBuildSliceFrom](
        s1: S1,
        s2: S2,
        s3: S3): Expr[A] = slice(Projection(s1, s2, s3))

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
      *
      * @param shape a new shape
      * @return an output with new shape
      */
    def reshape(shape: Shape): Expr[A] = f.reshape(expr, shape)
    def reshape(dim1: Int): Expr[A] = reshape(Shape(dim1))
    def reshape(dim1: Int, dim2: Int): Expr[A] = reshape(Shape(dim1, dim2))
    def reshape(dim1: Int, dim2: Int, dim3: Int): Expr[A] = reshape(Shape(dim1, dim2, dim3))
    def reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): Expr[A] = reshape(Shape(dim1, dim2, dim3, dim4))

    /** Removes dimensions of size 1 from the shape of a tensor.
      *
      * Given a tensor, this operation returns a tensor of the same type with all dimensions of size 1 removed.
      *
      * @return squeezed output
      */
    def squeeze: Expr[A] = f.squeeze(expr)

    def joinAlong(other: Expr[A], dim: Int): Expr[A] = f.joinAlong(Seq(expr, other), dim)
    def join(other: Expr[A]): Expr[A] = joinAlong(other, 0)

    def zip(second: Expr[A]): Expr[A] = f.zip(expr, second)

    def unzip: (Expr[A], Expr[A]) = {
      require(expr.shape.rank > 0, "cannot unzip a scalar")
      require(
        expr.shape.head == 2,
        s"first dimension should be equal to 2 but was ${expr.shape.head}")
      unzipN match {
        case Seq(first, second) => (first, second)
      }
    }

    def unzip3: (Expr[A], Expr[A], Expr[A]) = {
      require(expr.shape.rank > 0, "cannot unzip a scalar")
      require(
        expr.shape.head == 3,
        s"first dimension should be equal to 3 but was ${expr.shape.head}")
      unzipN match {
        case Seq(first, second, third) => (first, second, third)
      }
    }

    def unzipN: Seq[Expr[A]] = {
      require(expr.shape.rank > 0, "cannot unzip a scalar")
      val shape = expr.shape.remove(0)
      (0 until expr.shape.head).map(i => f.reshape(f.slice(expr, Projection(i)), shape))
    }

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
    def dependsOn(dep: Expr[_]): Expr[A] = f.dependsOn(expr, dep)
    def <<(dep: Expr[_]): Expr[A] = f.dependsOn(expr, dep)

    /** Cast elements of given tensor form type A into B.
      * Returns given input if A is already equal to B.
      *
      * @return casted output
      */
    def cast[B: TensorType]: Expr[B] = f.cast[A, B](expr)

    /** Return current op as leaf operation - means it doesnt produce output
      * and can be evaluated in graph only if added as dependant operation.
      *
      * @see dependsOn
      * @return current leaf node
      */
    def asVoid: Expr[Nothing] = expr.asInstanceOf[Expr[Nothing]]
  }

  class AnyOps(expr: Expr[_]) {
    import scanet.core.{kernels => f}

    /** Cast elements of given tensor form any type into B.
      * That is unsafe operation and if cast is impossible will fail in runtime
      *
      * @return casted output
      */
    def castUnsafe[B: TensorType]: Expr[B] = f.castUnsafe(expr)
  }

  trait AllSyntax extends AllKernels {
    implicit def toCoreKernelTensorTypeOps[A: TensorType](expr: Expr[A]): TensorTypeOps[A] =
      new TensorTypeOps[A](expr)
    implicit def toCoreKernelAnyOps[A](expr: Expr[A]): AnyOps =
      new AnyOps(expr)
  }

  object syntax extends AllSyntax
}
