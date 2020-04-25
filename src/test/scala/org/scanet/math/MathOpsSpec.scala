package org.scanet.math

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._

import scala.reflect.io.Path._

class MathOpsSpec extends AnyFlatSpec with Matchers {

  "plus" should "add 2 scalars" in {
    (2.0f.const plus 5.0f.const).eval should be(Tensor.scalar(7.0f))
  }

  "plus" should "add 2 tensors when one includes shape of the other" in {
    val result = Tensor.matrix(Array(1, 2), Array(1, 2)).const plus Tensor.vector(1, 2).const
    result.eval should be(Tensor.matrix(Array(2, 4), Array(2, 4)))
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
    val a = Tensor.scalar(1.0f).const.as("a")
    val b = Tensor.scalar(1.0f).const.as("b")
    val c = (a plus b).as("c")
    c.display()
    // run tensor-board: tensorboard --logdir .
    val files = ".".toDirectory.files.map(_.path).filter(_ matches ".*events.out.tfevents.*")
    files should not be empty
    files.foreach(_.toFile.delete())
  }
}
