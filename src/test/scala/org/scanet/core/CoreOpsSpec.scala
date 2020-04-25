package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.syntax._

class CoreOpsSpec extends AnyFlatSpec with Matchers {

  "const" should "be evaluated" in {
    5.0f.const.eval should be(Tensor.scalar(5.0f))
  }

  "evaluated tensor" should "be specialized" in {
    // todo: make const(5.0f)).eval specialized
    // this works: println(OpEval(const(5.0f)).eval.getClass)
    // this does not: println(const(5.0f).eval.getClass)
    // might need to specialize Op, but that is too much overhead, try to avoid that
    // however, we only really care about Tensor.toArray() to return primitive array which works anyway
    val tensor: Tensor[Float] = Session.run(5.0f.const)
    tensor.getClass.getName should endWith("Tensor$mcF$sp")
  }

  "product of 2 ops" should "be evaluated" in {
    (1.const, 2.const).eval should be((Tensor.scalar(1), Tensor.scalar(2)))
  }

  "product of 3 ops" should "be evaluated" in {
    (1.const, 2.const, 3.const).eval should be((Tensor.scalar(1), Tensor.scalar(2), Tensor.scalar(3)))
  }
}
