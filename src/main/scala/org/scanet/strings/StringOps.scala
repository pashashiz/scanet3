package org.scanet.strings

import org.scanet.core.TensorType.syntax._
import org.scanet.core.{Output, Shape, TensorType}
import simulacrum.typeclass

import scala.Ordering.Implicits._
import scala.language.higherKinds

@typeclass trait StringOps[F[_]] {

  /** Print current tensor during graph evaluation with default destination and formatting options.
   *
   * {{{Tensor.vector(1, 2).const.print.eval should be(Tensor.vector(1, 2))}}}
   *
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A]): F[A] = print(op, PrintOps())

  /** Print current tensor during graph evaluation into specified file with given formatting options.
   *
   * {{{Tensor.vector(1, 2).const.print("tensor.log").eval should be(Tensor.vector(1, 2))}}}
   *
   * @param file name to write (append) current tensor
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A], file: String): F[A] = print(op, PrintOps(dst = ToFile(file)))

  /** Print current tensor during graph evaluation into specified
   * output (i.e. stdout or a file) with given formatting options.
   *
   * {{{Tensor.vector(1, 2).const.print(dst = LogInfo).eval should be(Tensor.vector(1, 2))}}}
   *
   * @param ops output (destination, delimiter) and formatting options
   * @return current tensor (with print side effect)
   */
  def print[A: TensorType](op: F[A], ops: PrintOps): F[A]

  /** Format summary of given tensor into scalar string with default options.
   *
   * {{{Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))}}}
   *
   * @param op tensor to format
   * @return formatted tensor
   */
  def format[A: TensorType](op: F[A]): F[String] = format(op, FormatOps())

  /** Format summary of given tensor into scalar string.
   *
   * Formatting can be parametrised with template string, placeholder value (for current tensor)
   * and tensor summarize (number of leading and trailing tensor rows to print).
   *
   * {{{Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))}}}
   *
   * @param op  tensor to format
   * @param ops formatting options
   * @return formatted tensor
   */
  def format[A: TensorType](op: F[A], ops: FormatOps): F[String]

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

  trait Syntax extends Instances with StringOps.ToStringOpsOps {

    /** Joins the strings in the given list of string tensors into one tensor with given separator
     *
     * {{{join(",", "a".const, "b".const, "c".const).eval should be("a,b,c".const)}}}
     *
     * @param sep elements separator
     * @param ops list of string tensors to join
     * @return single string tensor
     */
    def join(sep: String, ops: Output[String]*): Output[String] = {
      require(ops.zip(ops.tail).forall({ case (o1, o2) => o1.broadcastableAny(o2) }),
        s"all tensors should have broadcastable shapes ${ops.map(_.shape)}")
      Output.name[String]("StringJoin")
        .shape(ops.map(_.shape).max)
        .inputs(ops: _*)
        .compileWithAttr("separator", sep)
        .compileWithInputList
        .build
    }
  }

  object syntax extends Syntax

}

class OutputStringOps extends StringOps[Output] {

  override def print[A: TensorType](op: Output[A], ops: PrintOps): Output[A] = {
    // format input op into scalar string
    val formatted = format(op, ops.formatOps)
    // print op doesn't have output - so we attach it as control op into next operation
    val printOp = Output.name[String]("PrintV2")
      .label(op.id)
      .shape(op.shape)
      .inputs(formatted)
      .compileWithAttr("output_stream", ops.dst.name)
      .compileWithAttr("end", ops.sep)
      .compileWithAllInputs
      .build
    // add identity op to attach print op and return origin output for chaining
    Output.name[A]("Identity")
      .shape(op.shape)
      .inputs(op)
      .compileWithAllInputs
      .compileWithControlOp(printOp)
      .build
  }

  override def format[A: TensorType](op: Output[A], ops: FormatOps): Output[String] = {
    Output.name[String]("StringFormat")
      .label(op.id)
      .shape(Shape())
      .inputs(op)
      .compileWithAttr("template", ops.template)
      .compileWithAttr("placeholder", ops.placeholder)
      .compileWithAttr("summarize", ops.summarize)
      .compileWithInputList
      .build
  }

  override def asString[A: TensorType](op: Output[A]): Output[String] = {
    if (TensorType[A] == TensorType[String]) op.asInstanceOf[Output[String]]
    else Output.name[String]("AsString")
      .shape(op.shape)
      .inputs(op)
      .compileWithAllInputs
      .build
  }

  override def concat[A: TensorType : Textual](left: Output[A], right: Output[A]): Output[A] = {
    require(left.shape.broadcastableAny(right.shape),
      s"cannot join tensors with shapes ${left.shape} + ${right.shape}")
    Output.name[A]("StringJoin")
      .shape(left.shape max right.shape)
      .inputs(left, right)
      .compileWithInputList
      .build
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

case class FormatOps(template: String = null, placeholder: String = null, summarize: Int = 3)
case class PrintOps(dst: PrintTo = StdErr, formatOps: FormatOps = FormatOps(), sep: String = null)

sealed abstract class PrintTo(val name: String)
case object LogInfo extends PrintTo("log(info)")
case object LogWarn extends PrintTo("log(warning)")
case object LogErr extends PrintTo("log(error)")
case object StdErr extends PrintTo("stderr")
case object StdOut extends PrintTo("stdout")
case class ToFile(path: String) extends PrintTo("file://" + path)