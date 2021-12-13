package scanet.strings

import scanet.core.DefaultCompiler.Ctx
import scanet.core.TensorType.syntax._
import scanet.core.{Compiler, DefaultCompiler, Expr, Shape, TensorType, Textual}
import org.tensorflow.OperationBuilder

import scala.collection.immutable.Seq

sealed abstract class PrintTo(val name: String)
case object LogInfo extends PrintTo("log(info)")
case object LogWarn extends PrintTo("log(warning)")
case object LogErr extends PrintTo("log(error)")
case object StdErr extends PrintTo("stderr")
case object StdOut extends PrintTo("stdout")
case class ToFile(path: String) extends PrintTo("file://" + path)

case class AsString[A: TensorType] private (expr: Expr[A]) extends Expr[String] {
  override def name: String = "AsString"
  override def tpe: Option[TensorType[String]] = Some(TensorType[String])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[String] = DefaultCompiler[String]()
}

object AsString {
  def apply[A: TensorType](expr: Expr[A]): Expr[String] =
    if (TensorType[A].tag == TensorType[String].tag)
      expr.asInstanceOf[Expr[String]]
    else
      new AsString(expr)
}

case class StringFormat[A: TensorType] private (tpl: String, expr: Seq[Expr[A]])
    extends Expr[String] {
  override def name: String = "StringFormat"
  override def tpe: Option[TensorType[String]] = Some(TensorType[String])
  override def shape: Shape = Shape.scalar
  override def inputs: Seq[Expr[_]] = expr
  override def compiler: Compiler[String] = DefaultCompiler[String](inputsAsList = true)
    .withAttrs(if (tpl.nonEmpty) Map("template" -> tpl, "placeholder" -> "{}") else Map())
}

object StringFormat {
  def apply[A: TensorType](tpl: String, expr: Seq[Expr[A]]): Expr[String] =
    // if template is empty and given single scalar value - just convert it to string
    if (tpl.isEmpty && expr.size == 1 && expr.head.shape.isScalar)
      AsString(expr.head)
    else
      new StringFormat(tpl, expr)
}

case class Print(dst: PrintTo, expr: Expr[String]) extends Expr[Nothing] {
  override def name: String = "PrintV2"
  override def tpe: Option[TensorType[Nothing]] = None
  override def shape: Shape = Shape.scalar
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[Nothing] =
    DefaultCompiler[Nothing]().withAttr("output_stream", dst.name)
}

case class AssertThat(cond: Expr[Boolean], expr: Expr[String]) extends Expr[Nothing] {
  override def name: String = "Assert"
  override def tpe: Option[TensorType[Nothing]] = None
  override def shape: Shape = Shape.scalar
  override def inputs: Seq[Expr[_]] = Seq(cond, expr)
  override def compiler: Compiler[Nothing] =
    DefaultCompiler[Nothing](withInputs = false).withStage {
      (ctx: Ctx, builder: OperationBuilder) =>
        builder.addInput(ctx.inputs.head.outputOrFail)
        builder.addInputList(ctx.inputs.tail.map(_.outputOrFail).toArray)
    }
}

case class StringJoin[A: Textual](sep: String, expr: Seq[Expr[A]]) extends Expr[A] {
  require(
    expr.zip(expr.tail).forall({ case (o1, o2) => o1.broadcastableAny(o2) }),
    s"all tensors should have broadcastable shapes ${expr.map(_.shape)}")
  override def name: String = "StringJoin"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override val shape: Shape = expr.map(_.shape).max
  override def inputs: Seq[Expr[_]] = expr
  override def compiler: Compiler[A] =
    DefaultCompiler[A](inputsAsList = true).withAttr("separator", sep)
}

case class StringLength[A: Textual](expr: Expr[A]) extends Expr[Int] {
  override def name: String = "StringLength"
  override def tpe: Option[TensorType[Int]] = Some(TensorType[Int])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[Int] = DefaultCompiler[Int]()
}

case class StringToNumber[A: Textual, B: TensorType](expr: Expr[A]) extends Expr[B] {
  override def name: String = "StringToNumber"
  override def tpe: Option[TensorType[B]] = Some(TensorType[B])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr)
  override def compiler: Compiler[B] = DefaultCompiler[B]().withAttr("out_type", TensorType[B])
}

case class Substring[A: Textual](expr: Expr[A], pos: Expr[Int], len: Expr[Int]) extends Expr[A] {
  override def name: String = "Substr"
  override def tpe: Option[TensorType[A]] = Some(TensorType[A])
  override def shape: Shape = expr.shape
  override def inputs: Seq[Expr[_]] = Seq(expr, pos, len)
  override def compiler: Compiler[A] = DefaultCompiler[A]()
}

trait AllKernels {

  /** Converts elements of given tensor into Strings.
    * Returns given input if tensor already contains Strings.
    *
    * {{{ asString(Tensor.vector(1, 2, 3).const).eval should be(Tensor.vector("1", "2", "3")) }}}
    *
    * @return output converted to strings
    */
  def asString[A: TensorType](expr: Expr[A]): Expr[String] = AsString(expr)

  /** Format given tensors with specified template.
    *
    * {{{format("vector: {}", Tensor.vector(1, 2, 3) should be(Tensor.scalar("[1 2 3]"))}}}
    *
    * @param tpl string with `{}` placeholder for current tensor
    * @param expr tensors to format with given template
    * @return formatted scalar string
    */
  def format[A: TensorType](tpl: String, expr: Expr[A]*): Expr[String] =
    StringFormat(tpl, expr.toList)

  /** Print given tensors formatted with given template during graph evaluation.
    *
    * {{{
    * val a = 1.const
    * val b = 2.const
    * val c = (a plus b) dependsOn print("a + b = {} + {}", a, b)
    * c.eval should be(Tensor.scalar(3)) // and prints `a + b = 1 + 2`
    * }}}
    *
    * @param tpl message with `{}` placeholder for each of given tensors
    * @param expr ensors to format and print
    * @return leaf output
    */
  def print[A: TensorType](tpl: String, expr: Expr[A]*): Expr[Nothing] =
    print(LogInfo, tpl, expr: _*)

  /** Print given tensors formatted with given template during graph evaluation into specified location
    *
    * {{{
    * val a = 1.const
    * val b = 2.const
    * val c = (a plus b) dependsOn print(ToFile("temp.txt"), "a + b = {} + {}", a, b)
    * c.eval should be(Tensor.scalar(3)) // and prints `a + b = 1 + 2`
    * }}}
    *
    * @param dst f print stream (i.e. stdout, log info, or a file)
    * @param tpl message with `{}` placeholder for each of given tensors
    * @param expr tensors to format and print
    * @return leaf output
    */
  def print[A: TensorType](dst: PrintTo, tpl: String, expr: Expr[A]*): Expr[Nothing] =
    Print(dst, format(tpl, expr: _*))

  /** Asserts that given condition is true or fail graph execution.
    * Optional outputs can be specified to print on assertion failure.
    *
    * {{{
    * val a = 2.const
    * val b = 1.const
    * val c = (a div b) << assertThat(a gt b, "{} {}", a, b)
    * c.eval should be(Tensor.scalar(2))
    * }}}
    *
    * @param cond  assertion condition
    * @param tpl message with `{}` placeholder for current op
    * @param expr outputs to format into error message
    * @see dependsOn
    * @return leaf node to add as dependant op
    */
  def assertThat[A: TensorType](cond: Expr[Boolean], tpl: String, expr: Expr[A]*): Expr[Nothing] =
    AssertThat(cond, format(tpl, expr: _*))

  /** Joins the strings in the given list of string tensors into one tensor with given separator.
    *
    * {{{join(",", "a".const, "b".const, "c".const).eval should be("a,b,c".const)}}}
    *
    * @param sep elements separator
    * @param expr list of string tensors to join
    * @return single string tensor
    */
  def join[A: Textual](sep: String, expr: Expr[A]*): Expr[A] =
    StringJoin[A](sep, expr.toList)
}

object kernels extends AllKernels {

  class AnyOps[A: TensorType](expr: Expr[A]) {
    import scanet.core.kernels.syntax._
    import scanet.strings.{kernels => f}

    /** Print current tensor during graph evaluation into default location.
      *
      * {{{Tensor.vector(1, 2).const.print.eval should be(Tensor.vector(1, 2))}}}
      *
      * @return current tensor (with print side effect)
      */
    def print: Expr[A] = print(template = "")

    /** Print current tensor during graph evaluation into specified output (i.e. stdout or a file).
      *
      * {{{Tensor.vector(1, 2).const.print(LogInfo).eval should be(Tensor.vector(1, 2))}}}
      *
      * @return current tensor (with print side effect)
      */
    def print(dst: PrintTo): Expr[A] = print(dst, template = "")

    /** Print formatted current tensor during graph evaluation into default location.
      *
      * {{{Tensor.vector(1, 2).const.print("vector: {}").eval should be(Tensor.vector(1, 2))}}}
      *
      * @param template message with `{}` placeholder for current tensor
      * @return current tensor (with print side effect)
      */
    def print(template: String): Expr[A] = print(LogInfo, template)

    /** Print formatted current tensor during graph evaluation into specified output (i.e. stdout or a file).
      *
      * {{{Tensor.vector(1, 2).const.print(LogWarn, "vector: {}").eval should be(Tensor.vector(1, 2))}}}
      *
      * @param template message with `{}` placeholder for current tensor
      * @param dst      of print stream (i.e. stdout, log info, or a file)
      * @return current tensor (with print side effect)
      */
    def print(dst: PrintTo, template: String): Expr[A] =
      expr << f.print(dst, template, expr)

    /** Asserts that given condition (constructed from current tensor by specified fn)
      * is true or fail graph execution (and prints current value).
      *
      * {{{
      * val a = Tensor.vector(1, 2).const
      * val b = Tensor.vector(3, 4).const
      * val c = Tensor.vector(4, 6).const
      *
      * (a plus b).assert(_ === c, "sum was {}").eval should be(Tensor.vector(4, 6))
      * }}}
      *
      * @param f function to build assertion condition from current op
      * @param template message with `{}` placeholder for current op
      * @return current op for chaining
      */
    def assert(f: Expr[A] => Expr[Boolean], template: String): Expr[A] =
      expr << assertThat(f(expr), template, expr)

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
    def assert(f: Expr[A] => Expr[Boolean]): Expr[A] = assert(f, "value: {}")

    /** Format summary of given tensor into scalar string.
      *
      * {{{Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))}}}
      *
      * @return formatted tensor
      */
    def format: Expr[String] = format(template = "")

    /** Format summary of given tensor into scalar string using template string with `{}` placeholder.
      *
      * {{{Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))}}}
      *
      * @param template string with `{}` placeholder for current tensor
      * @return formatted tensor
      */
    def format(template: String): Expr[String] = f.format(template, expr)

    /** Converts elements of given tensor into Strings.
      * Returns given input if tensor already contains Strings.
      *
      * {{{ Tensor.vector(1, 2, 3).const.asString.eval should be(Tensor.vector("1", "2", "3")) }}}
      *
      * @return output converted to strings
      */
    def asString: Expr[String] = f.asString(expr)
  }

  class TextualOps[A: Textual](expr: Expr[A]) {
    import scanet.strings.{kernels => f}

    /** Concatenates current and given String tensors
      *
      * {{{Tensor.vector("ab", "cd").const.concat("e".const).eval should be(Tensor.vector("abe", "cde")}}}
      *
      * @param right side to append
      * @return tensors containing joined corresponding strings of left and right tensors
      */
    def concat(right: Expr[A]): Expr[A] = {
      require(
        expr.shape.broadcastableAny(right.shape),
        s"cannot join tensors with shapes ${expr.shape} + ${right.shape}")
      f.join("", expr, right)
    }

    /** Computes the length of each string given in the input tensor.
      *
      * {{{Tensor.vector("a", "bb").const.length.eval should be(Tensor.vector(1, 2))}}}
      *
      * @return tensor with strings lengths
      */
    def length: Expr[Int] = StringLength(expr)

    /** Converts each string in the input Tensor to the specified numeric type.
      *
      * {{{Tensor.vector("1.1", "2.2").const.toNumber[Float].eval should be(Tensor.vector(1.1f, 2.2f))}}}
      *
      * @tparam B data type for output tensor
      * @return tensor with parsed numbers
      */
    def toNumber[B: TensorType]: Expr[B] = StringToNumber(expr)

    /** Return substrings from tensor of strings.
      *
      * For each string in the input Tensor, creates a substring starting at index
      * pos with a total length of len.
      *
      * If len defines a substring that would extend beyond the length of the input
      * string, then as many characters as possible are used.
      *
      * A negative pos indicates distance within the string backwards from the end.
      *
      * If pos specifies an index which is out of range for any of the input strings,
      * then an InvalidArgumentError is thrown.
      *
      * pos and len must have the same shape and supports broadcasting up to two dimensions
      *
      * {{{"abcde".const.substring(1.const, 3.const).eval should be("bcd".const)}}}
      *
      * @param pos - tensor of substring starting positions
      * @param len - tensor of substrings lengths
      * @return tensor of substrings
      */
    def substring(pos: Expr[Int], len: Expr[Int]): Expr[A] = Substring(expr, pos, len)
  }

  trait AllSyntax extends AllKernels {
    implicit def toStringKernelOps[A: TensorType](expr: Expr[A]): AnyOps[A] =
      new AnyOps[A](expr)
    implicit def toStringKernelTextualOps[A: Textual](expr: Expr[A]): TextualOps[A] =
      new TextualOps[A](expr)
  }

  object syntax extends AllSyntax
}
