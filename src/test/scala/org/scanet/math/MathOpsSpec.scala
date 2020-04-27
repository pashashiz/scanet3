package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.{Shape, Tensor}
import org.scanet.math.syntax._

import scala.reflect.io.Path._

class MathOpsSpec extends AnyFlatSpec with Matchers {

  "plus" should "add 2 scalars" in {
    (2.0f.const plus 5.0f.const).eval should be(Tensor.scalar(7.0f))
  }

  "plus" should "add 2 tensors when one includes shape of the other" in {
    val a = Tensor.matrix(
      Array(1, 2),
      Array(1, 2))
    val b = Tensor.vector(1, 2)
    val c = Tensor.matrix(
      Array(2, 4),
      Array(2, 4))
    (a.const plus b.const).eval should be(c)
  }

  "plus" should "work when adding same tensor" in {
    val a = 5.0f.const.as("a")
    (a plus a).as("c").eval should be(Tensor.scalar(10.0f))
  }

  "plus" should "fail when 2 tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.matrix(Array(1, 2), Array(1, 2)).const plus Tensor.vector(1, 2, 3).const
    } should have message "requirement failed: tensors with shapes (2, 2) and (3) cannot be added, " +
      "one of the tensors should have shape which includes the other"
  }

  "computation graph" should "be displayed" in {
    val a = 1.0f.const.as("a")
    val b = 1.0f.const.as("b")
    val c = (a plus b).as("c")
    c.display()
    // run tensor-board: tensorboard --logdir .
    val files = ".".toDirectory.files.map(_.path).filter(_ matches ".*events.out.tfevents.*")
    files should not be empty
    files.foreach(_.toFile.delete())
  }

  "multiply" should "produce dot product on 2 matrices" in {
    val a = Tensor.matrix(
      Array(1, 2, 3),
      Array(1, 2, 3))
    val b = Tensor.matrix(
      Array(1, 2),
      Array(1, 2),
      Array(1, 2))
    val c = Tensor.matrix(
      Array(6, 12),
      Array(6, 12))
    (a.const * b.const).eval should be(c)
  }

  "multiply" should "produce dot product of vector and matrix" in {
    val a = Tensor.vector(1, 2, 3)
    val b = Tensor.matrix(
      Array(1, 2),
      Array(1, 2),
      Array(1, 2))
    (a.const * b.const).eval should be(Tensor.vector(6, 12))
  }

  "multiply" should "multiply 2 scalars" in {
    (2.const * 3.const).eval should be(Tensor.scalar(6))
  }

  "multiply" should "multiply scalar and vector" in {
    (2.const * Tensor.vector(1, 2, 3).const).eval should be(Tensor.vector(2, 4, 6))
  }

  "multiply" should "fail to multiply vector and scalar" in {
    the [IllegalArgumentException] thrownBy {
      (Tensor.vector(1, 2, 3).const * 2.const).eval
    } should have message "requirement failed: cannot multiply tensors with shapes (1, 3) * (1, 1)"
  }

  "multiply" should "fail to multiply 3D tensors" in {
    the [IllegalArgumentException] thrownBy {
      val tensor = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2))
      (tensor.const * tensor.const).eval
    } should have message "requirement failed: rank cannot be > 2 but got tensors with shapes (2, 2, 2) * (2, 2, 2)"
  }
}