package org.scanet.strings

import org.scanet.core.TensorType.syntax._
import org.scanet.core.{Output, TensorType, TfTypeString}
import simulacrum.typeclass

import scala.Ordering.Implicits._
import scala.language.higherKinds

@typeclass trait StringOps[F[_]] {

  /** Concatenates current and given String tensors
   *
   * {{{Tensor.vector("ab", "cd").const.concat("e".const).eval should be(Tensor.vector("abe", "cde")}}}
   *
   * @param left  side
   * @param right side to append
   * @return tensors containing joined corresponding strings of left and right tensors
   */
  def concat[A: TensorType : StringLike](left: F[A], right: F[A]): F[A]

  /** Computes the length of each string given in the input tensor.
   *
   * {{{Tensor.vector("a", "bb").const.length.eval should be(Tensor.vector(1, 2))}}}
   *
   * @return tensor with strings lengths
   */
  def length[A: TensorType : StringLike](op: F[A]): F[Int]

  /** Converts each string in the input Tensor to the specified numeric type.
   *
   * {{{Tensor.vector("1.1", "2.2").const.toNumber[Float].eval should be(Tensor.vector(1.1f, 2.2f))}}}
   *
   * @tparam B data type for output tensor
   * @return tensor with parsed numbers
   */
  def toNumber[A: TensorType : StringLike, B: TensorType](op: F[A]): F[B]

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
  def substring[A: TensorType : StringLike](op: F[A], pos: F[Int], len: F[Int]): F[A]

  //  def split[A: TensorType : StringLike](op: F[A], sep: F[String]): F[String]
}

trait StringLike[S]

object StringOps {

  trait Instances {
    implicit def outputStringOps: StringOps[Output] = new OutputStringOps

    implicit def stringLikeInst: StringLike[String] = new StringLike[String] with TfTypeString
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

  override def concat[A: TensorType : StringLike](left: Output[A], right: Output[A]): Output[A] = {
    require(left.shape.broadcastableAny(right.shape),
      s"cannot join tensors with shapes ${left.shape} + ${right.shape}")
    Output.name[A]("StringJoin")
      .shape(left.shape max right.shape)
      .inputs(left, right)
      .compileWithInputList
      .build
  }

  override def length[A: TensorType : StringLike](op: Output[A]): Output[Int] = {
    Output.name[Int]("StringLength")
      .shape(op.shape)
      .inputs(op)
      .compileWithAllInputs
      .build
  }

  override def toNumber[A: TensorType : StringLike, B: TensorType](op: Output[A]): Output[B] = {
    Output.name[B]("StringToNumber")
      .shape(op.shape)
      .inputs(op)
      .compileWithAttr("out_type", TensorType[B])
      .compileWithAllInputs
      .build
  }

  override def substring[A: TensorType : StringLike](op: Output[A], pos: Output[Int], len: Output[Int]): Output[A] = {
    require(pos.shape == len.shape, s"pos and len shapes are not equal ${pos.shape}, ${len.shape}")
    require(op.shape.broadcastableBy(pos.shape),
      s"string tensor shape (${op.shape}) is not broadcastable by len/pos shape ${pos.shape}")
    Output.name[A]("Substr")
      .shape(op.shape)
      .inputs(op, pos, len)
      .compileWithAllInputs
      .build
  }

  //  override def split[A: TensorType : StringLike](op: Output[A], sep: Output[String]): Output[String] = {
  //    Output.name[String]("StringSplitV2")
  //      .shape(op.shape) // TODO: this is wrong, result is SparseTensor with different shape
  //      .inputs(op, sep)
  //      .compileWithAllInputs
  //      .build
  //  }
}