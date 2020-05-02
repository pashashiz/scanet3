package org.scanet.strings

import org.scanet.core.syntax._
import org.scanet.core.{Output, Shape, TensorType}
import simulacrum.typeclass

import scala.language.higherKinds

@typeclass trait StringOps[F[_]] {

  /** Print current tensor during graph evaluation into default location.
   *
   * {{{Tensor.vector(1, 2).const.print.eval should be(Tensor.vector(1, 2))}}}
   *
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A]): F[A] = print(op, template = "")

  /** Print current tensor during graph evaluation into specified output (i.e. stdout or a file).
   *
   * {{{Tensor.vector(1, 2).const.print(LogInfo).eval should be(Tensor.vector(1, 2))}}}
   *
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A], dst: PrintTo): F[A] = print(op, dst, template = "")

  /** Print formatted current tensor during graph evaluation into default location.
   *
   * {{{Tensor.vector(1, 2).const.print("vector: {}").eval should be(Tensor.vector(1, 2))}}}
   *
   * @param template message with `{}` placeholder for current tensor
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A], template: String): F[A] = print(op, LogInfo, template)

  /** Print formatted current tensor during graph evaluation into specified output (i.e. stdout or a file).
   *
   * {{{Tensor.vector(1, 2).const.print(LogWarn, "vector: {}").eval should be(Tensor.vector(1, 2))}}}
   *
   * @param template message with `{}` placeholder for current tensor
   * @param dst      of print stream (i.e. stdout, log info, or a file)
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A], dst: PrintTo, template: String): F[A]

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
  def assert[A: TensorType](op: F[A], f: F[A] => F[Boolean], template: String): F[A]

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
  def assert[A: TensorType](op: F[A], f: F[A] => F[Boolean]): F[A] = assert(op, f, "value: {}")

  /** Format summary of given tensor into scalar string.
   *
   * {{{Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))}}}
   *
   * @param op tensor to format
   * @return formatted tensor
   */
  def format[A: TensorType](op: F[A]): F[String] = format(op, template = "")

  /** Format summary of given tensor into scalar string using template string with `{}` placeholder.
   *
   * {{{Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))}}}
   *
   * @param op       tensor to format
   * @param template string with `{}` placeholder for current tensor
   * @return formatted tensor
   */
  def format[A: TensorType](op: F[A], template: String): F[String]

  /** Converts elements of given tensor into Strings.
   * Returns given input if tensor already contains Strings.
   *
   * {{{ Tensor.vector(1, 2, 3).const.asString.eval should be(Tensor.vector("1", "2", "3")) }}}
   *
   * @return output converted to strings
   */
  def asString[A: TensorType](op: F[A]): F[String]

  /** Concatenates current and given String tensors
   *
   * {{{Tensor.vector("ab", "cd").const.concat("e".const).eval should be(Tensor.vector("abe", "cde")}}}
   *
   * @param left  side
   * @param right side to append
   * @return tensors containing joined corresponding strings of left and right tensors
   */
  def concat[A: TensorType : Textual](left: F[A], right: F[A]): F[A]

  /** Computes the length of each string given in the input tensor.
   *
   * {{{Tensor.vector("a", "bb").const.length.eval should be(Tensor.vector(1, 2))}}}
   *
   * @return tensor with strings lengths
   */
  def length[A: TensorType : Textual](op: F[A]): F[Int]

  /** Converts each string in the input Tensor to the specified numeric type.
   *
   * {{{Tensor.vector("1.1", "2.2").const.toNumber[Float].eval should be(Tensor.vector(1.1f, 2.2f))}}}
   *
   * @tparam B data type for output tensor
   * @return tensor with parsed numbers
   */
  def toNumber[A: TensorType : Textual, B: TensorType](op: F[A]): F[B]

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
  def substring[A: TensorType : Textual](op: F[A], pos: F[Int], len: F[Int]): F[A]
}

trait Textual[S]

object StringOps {

  trait Instances {
    implicit def outputStringOps: StringOps[Output] = new OutputStringOps

    implicit def stringIsTextual: Textual[String] = new Textual[String] {}
  }

  trait Syntax extends Instances with StringOps.ToStringOpsOps with OutputStringMultiOps

  object syntax extends Syntax

}

trait OutputStringMultiOps {

  /** Print given tensors formatted with given template during graph evaluation.
   *
   * {{{
   * val a = 1.const
   * val b = 2.const
   * val c = (a plus b) dependsOn print("a + b = {} + {}", a, b)
   * c.eval should be(Tensor.scalar(3)) // and prints `a + b = 1 + 2`
   * }}}
   *
   * @param template message with `{}` placeholder for each of given tensors
   * @param ops      tensors to format and print
   * @return leaf output
   */
  def print[A: TensorType](template: String, ops: Output[A]*): Output[Nothing] = {
    print(LogInfo, template, ops: _*)
  }

  /** Print given tensors formatted with given template during graph evaluation into specified location
   *
   * {{{
   * val a = 1.const
   * val b = 2.const
   * val c = (a plus b) dependsOn print(ToFile("temp.txt"), "a + b = {} + {}", a, b)
   * c.eval should be(Tensor.scalar(3)) // and prints `a + b = 1 + 2`
   * }}}
   *
   * @param dst      of print stream (i.e. stdout, log info, or a file)
   * @param template message with `{}` placeholder for each of given tensors
   * @param ops      tensors to format and print
   * @return leaf output
   */
  def print[A: TensorType](dst: PrintTo, template: String, ops: Output[A]*): Output[Nothing] = {
    // format given tensors (noop if its already a scalar string and template is empty)
    val formatted = format(template, ops: _*)

    // print doesnt have output - so we cast it to Output[Nothing] not to use it as input
    Output.name[String]("PrintV2")
      .shape(Shape())
      .inputs(formatted)
      .compileWithAttr("output_stream", dst.name)
      .compileWithAllInputs
      .build
      .asVoid
  }

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
   * @param template message with `{}` placeholder for current op
   * @param ops outputs to format into error message
   * @see dependsOn
   * @return leaf node to add as dependant op
   */
  def assertThat[A: TensorType](cond: Output[Boolean], template: String, ops: Output[A]*): Output[Nothing] = {
    Output.name[Boolean]("Assert")
      .shape(Shape())
      .inputs(cond, format(template, ops: _*))
      .compileWithTransformer((ctx, builder) => {
        // add condition as input
        builder.addInput(ctx.inputs.head.output(0))
        // add formatted msg as value to print
        builder.addInputList(ctx.inputs.tail.map(_.output(0)).toArray)
      })
      .build
      .asVoid
  }

  /** Format given tensors with specified template.
   *
   * {{{format("vector: {}, Tensor.vector(1, 2, 3) should be(Tensor.scalar("[1 2 3]"))}}}
   *
   * @param tpl string with `{}` placeholder for current tensor
   * @param ops tensors to format with given template
   * @return formatted scalar string
   */
  def format[A: TensorType](tpl: String, ops: Output[A]*): Output[String] = {
    if (tpl.isEmpty && ops.size == 1 && ops.head.shape.isScalar) {
      // if template is empty and given single scalar value - just convert it to string
      asString(ops.head)
    } else {
      Output.name[String]("StringFormat")
        .shape(Shape())
        .inputs(ops: _*)
        .compileWithAttrs(if (!tpl.isEmpty) Map("template" -> tpl, "placeholder" -> "{}") else Map())
        .compileWithInputList
        .build
    }
  }

  /** Joins the strings in the given list of string tensors into one tensor with given separator.
   *
   * {{{join(",", "a".const, "b".const, "c".const).eval should be("a,b,c".const)}}}
   *
   * @param sep elements separator
   * @param ops list of string tensors to join
   * @return single string tensor
   */
  def join[A: TensorType : Textual](sep: String, ops: Output[A]*): Output[A] = {
    require(ops.zip(ops.tail).forall({ case (o1, o2) => o1.broadcastableAny(o2) }),
      s"all tensors should have broadcastable shapes ${ops.map(_.shape)}")
    Output.name[A]("StringJoin")
      .shape(ops.map(_.shape).max)
      .inputs(ops: _*)
      .compileWithAttr("separator", sep)
      .compileWithInputList
      .build
  }

  /** Converts elements of given tensor into Strings.
   * Returns given input if tensor already contains Strings.
   *
   * {{{ asString(Tensor.vector(1, 2, 3).const).eval should be(Tensor.vector("1", "2", "3")) }}}
   *
   * @return output converted to strings
   */
  def asString[A: TensorType](op: Output[A]): Output[String] = {
    if (TensorType[A] == TensorType[String]) {
      op.asInstanceOf[Output[String]]
    } else {
      Output.name[String]("AsString")
        .shape(op.shape)
        .inputs(op)
        .compileWithAllInputs
        .build
    }
  }
}

class OutputStringOps extends StringOps[Output] with OutputStringMultiOps {

  override def print[A: TensorType](op: Output[A], dst: PrintTo, template: String): Output[A] = {
    op dependsOn print(dst, template, op)
  }

  override def format[A: TensorType](op: Output[A], template: String): Output[String] = {
    format(template, op)
  }

  override def assert[A: TensorType](op: Output[A], f: Output[A] => Output[Boolean], template: String): Output[A] = {
    op dependsOn assertThat(f(op), template, op)
  }

  override def concat[A: TensorType : Textual](left: Output[A], right: Output[A]): Output[A] = {
    require(left.shape.broadcastableAny(right.shape),
      s"cannot join tensors with shapes ${left.shape} + ${right.shape}")
    join("", left, right)
  }

  override def length[A: TensorType : Textual](op: Output[A]): Output[Int] = {
    Output.name[Int]("StringLength")
      .shape(op.shape)
      .inputs(op)
      .compileWithAllInputs
      .build
  }

  override def toNumber[A: TensorType : Textual, B: TensorType](op: Output[A]): Output[B] = {
    Output.name[B]("StringToNumber")
      .shape(op.shape)
      .inputs(op)
      .compileWithAttr("out_type", TensorType[B])
      .compileWithAllInputs
      .build
  }

  override def substring[A: TensorType : Textual](op: Output[A], pos: Output[Int], len: Output[Int]): Output[A] = {
    require(pos.shape == len.shape, s"pos and len shapes are not equal ${pos.shape}, ${len.shape}")
    require(op.shape.broadcastableBy(pos.shape),
      s"string tensor shape (${op.shape}) is not broadcastable by len/pos shape ${pos.shape}")
    Output.name[A]("Substr")
      .shape(op.shape)
      .inputs(op, pos, len)
      .compileWithAllInputs
      .build
  }
}

sealed abstract class PrintTo(val name: String)
case object LogInfo extends PrintTo("log(info)")
case object LogWarn extends PrintTo("log(warning)")
case object LogErr extends PrintTo("log(error)")
case object StdErr extends PrintTo("stderr")
case object StdOut extends PrintTo("stdout")
case class ToFile(path: String) extends PrintTo("file://" + path)