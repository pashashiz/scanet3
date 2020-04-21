package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.syntax.core._
import org.scanet.core.Op._

class OpSpec extends AnyFlatSpec with Matchers {

  "const" should "be evaluated" in {
    const(5.0f).eval should be(Tensor.scalar(5.0f))
  }

  "evaluated tensor" should "be specialized" in {
    // todo: make const(5.0f)).eval specialized
    // this works: println(OpEval(const(5.0f)).eval.getClass)
    // this does not: println(const(5.0f).eval.getClass)
    // might need to specialize Op, but that is too much overhead, try to avoid that
    // however, we only really care about Tensor.toArray() to return primitive array which works anyway
    val tensor: Tensor[Float] = Session.run(const(5.0f))
    tensor.getClass.getName should endWith("Tensor$mcF$sp")
  }

  "plus" should "add 2 scalars" in {
    plus(const(2.0f), const(5.0f)).eval should be(Tensor.scalar(7.0f))
  }

  "plus" should "add 2 tensors when one includes shape of the other" in {
    plus(const(Tensor.matrix(Array(1, 2), Array(1, 2))),
         const(Tensor.vector(1, 2))).eval should
      be(Tensor.matrix(Array(2, 4), Array(2, 4)))
  }

  "plus" should "work when adding same tensor" in {
    val a = const(5.0f, "a")
    plus(a, a, "c").eval should be(Tensor.scalar(10.0f))
  }

  "plus" should "fail when 2 tensors have incompatible dimensions" in {
    the [IllegalArgumentException] thrownBy {
      plus(const(Tensor.matrix(Array(1, 2), Array(1, 2))),
        const(Tensor.vector(1, 2, 3))).eval
    } should have message "requirement failed: tensors with shapes (2, 2) and (3) cannot be added, " +
      "one of the tensors should have shape which includes the other"
  }

  "product of 2 ops" should "be evaluated" in {
    (const(1), const(2)).eval should be((Tensor.scalar(1), Tensor.scalar(2)))
  }

  "product of 3 ops" should "be evaluated" in {
    (const(1), const(2), const(3)).eval should be((Tensor.scalar(1), Tensor.scalar(2), Tensor.scalar(3)))
  }


}
