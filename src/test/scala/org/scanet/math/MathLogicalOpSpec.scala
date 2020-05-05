package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._

class MathLogicalOpSpec extends AnyFlatSpec with Matchers {

  "element-wise equality" should "work on tensors with same dimensions" in {
    (Tensor.vector(1, 2, 3).const :== Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(true, true, false))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const :== 1.const).eval should be(Tensor.vector(true, false, false))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const :== Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot check for equality tensors with shapes (2, 2) :== (3)"
  }

  "element-wise non-equality" should "work on tensors with same dimensions" in {
    (Tensor.vector(1, 2, 3).const :!= Tensor.vector(1, 2, 5).const).eval should be(Tensor.vector(false, false, true))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const :!= 1.const).eval should be(Tensor.vector(false, true, true))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.matrix(Array(1, 2), Array(1, 2)).const :!= Tensor.vector(1, 2, 3).const).eval
    } should have message "requirement failed: cannot check for equality tensors with shapes (2, 2) :== (3)"
  }

  "equality" should "return true when tensors are equal" in {
    (Tensor.vector(1, 2, 3).const === Tensor.vector(1, 2, 3).const).eval should be(Tensor.scalar(true))
  }

  it should "return false when tensors are not equal" in {
    (Tensor.vector(1, 2, 3).const === Tensor.vector(1, 2, 4).const).eval should be(Tensor.scalar(false))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const === 1.const).eval should be(Tensor.vector(true, false, false))
    } should have message "requirement failed: cannot check equality tensors with different shapes (3) === ()"
  }

  "non-equality" should "return false when tensors are equal" in {
    (Tensor.vector(1, 2, 3).const !== Tensor.vector(1, 2, 3).const).eval should be(Tensor.scalar(false))
  }

  it should "return true when tensors are not equal" in {
    (Tensor.vector(1, 2, 3).const !== Tensor.vector(1, 2, 4).const).eval should be(Tensor.scalar(true))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const !== 1.const).eval should be(Tensor.vector(true, false, false))
    } should have message "requirement failed: cannot check non-equality tensors with different shapes (3) !== ()"
  }

  "all" should "return true if all elements are true" in {
    Tensor.vector(true, true).const.all.eval should be(Tensor.scalar(true))
  }

  it should "return false if at least one element is false" in {
    Tensor.vector(true, false).const.all.eval should be(Tensor.scalar(false))
  }

  it should "support reducing along matrix columns" in {
    Tensor.matrix(Array(true, false), Array(true, true)).const.all(Seq(0)).eval should be(Tensor.vector(true, false))
  }

  it should "support reducing along matrix rows" in {
    Tensor.matrix(Array(true, false), Array(true, true)).const.all(Seq(1)).eval should be(Tensor.vector(false, true))
  }

  it should "fail when given axis is out of bound" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.matrix(Array(true, false), Array(true, true)).const.all(Seq(2)).eval
    } should have message "requirement failed: tensor with rank 2 does not have (2) axises"
  }

  "any" should "return true if at least one element is true" in {
    Tensor.vector(true, false).const.any.eval should be(Tensor.scalar(true))
  }

  it should "return false if all elements are false" in {
    Tensor.vector(false, false).const.any.eval should be(Tensor.scalar(false))
  }

  it should "support reducing along matrix columns" in {
    Tensor.matrix(Array(false, false), Array(true, true)).const.any(Seq(0)).eval should be(Tensor.vector(true, true))
  }

  it should "support reducing along matrix rows" in {
    Tensor.matrix(Array(false, false), Array(true, true)).const.any(Seq(1)).eval should be(Tensor.vector(false, true))
  }

  it should "fail when given axis is out of bound" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.matrix(Array(true, false), Array(true, true)).const.any(Seq(2)).eval
    } should have message "requirement failed: tensor with rank 2 does not have (2) axises"
  }

  "greater comparison" should "work element wise" in {
    (Tensor.vector(1, 2, 3).const > Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(false, false, true))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const > 1.const).eval should be(Tensor.vector(false, true, true))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const > Tensor.vector(2, 3).const).eval
    } should have message "requirement failed: cannot compare tensors with shapes (3) > (2)"
  }

  "greater or equal comparison" should "work element wise" in {
    (Tensor.vector(1, 2, 3).const >= Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(false, true, true))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const >= 1.const).eval should be(Tensor.vector(true, true, true))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const >= Tensor.vector(2, 3).const).eval
    } should have message "requirement failed: cannot compare tensors with shapes (3) >= (2)"
  }

  "less comparison" should "work element wise" in {
    (Tensor.vector(1, 2, 3).const < Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(true, false, false))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const < 1.const).eval should be(Tensor.vector(false, false, false))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const < Tensor.vector(2, 3).const).eval
    } should have message "requirement failed: cannot compare tensors with shapes (3) < (2)"
  }

  "less or equal comparison" should "work element wise" in {
    (Tensor.vector(1, 2, 3).const <= Tensor.vector(3, 2, 1).const).eval should be(Tensor.vector(true, true, false))
  }

  it should "support broadcasting" in {
    (Tensor.vector(1, 2, 3).const <= 1.const).eval should be(Tensor.vector(true, false, false))
  }

  it should "fail when tensors have different shapes" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const <= Tensor.vector(2, 3).const).eval
    } should have message "requirement failed: cannot compare tensors with shapes (3) <= (2)"
  }

  "logical and" should "work on tensors with same dimensions" in {
    (Tensor.vector(true, false).const && Tensor.vector(true, true).const).eval should be(Tensor.vector(true, false))
  }

  it should "support broadcasting" in {
    (Tensor.vector(true, false).const && Tensor.vector(true).const).eval should be(Tensor.vector(true, false))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(true, false).const && Tensor.vector(true, true, true).const).eval
    } should have message "requirement failed: cannot logically AND tensors with shapes (2) && (3)"
  }

  "logical or" should "work on tensors with same dimensions" in {
    (Tensor.vector(true, false).const || Tensor.vector(true, true).const).eval should be(Tensor.vector(true, true))
  }

  it should "support broadcasting" in {
    (Tensor.vector(true, false).const || Tensor.vector(true).const).eval should be(Tensor.vector(true, true))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(true, false).const || Tensor.vector(true, true, true).const).eval
    } should have message "requirement failed: cannot logically OR tensors with shapes (2) || (3)"
  }

  "logical not" should "work" in {
    Tensor.vector(true, false).const.not.eval should be(Tensor.vector(false, true))
  }

  "logical xor" should "work on tensors with same dimensions" in {
    (Tensor.vector(true, false).const ^ Tensor.vector(true, true).const).eval should be(Tensor.vector(false, true))
  }

  it should "support broadcasting" in {
    (Tensor.vector(true, false).const ^ Tensor.vector(true).const).eval should be(Tensor.vector(false, true))
  }

  it should "fail when tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(true, false).const ^ Tensor.vector(true, true, true).const).eval
    } should have message "requirement failed: cannot logically XOR tensors with shapes (2) ^ (3)"
  }
}