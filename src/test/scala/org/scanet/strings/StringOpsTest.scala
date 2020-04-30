package org.scanet.strings

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.strings.syntax._

class StringOpsTest extends AnyFlatSpec with Matchers {

  "concat" should "concatenate two string vectors" in {
    val a = Tensor.vector("1", "2").const
    val b = Tensor.vector("3", "4").const

    a.concat(b).eval should be(Tensor.vector("13", "24"))
  }

  "concat" should "concatenate with scalar" in {
    val a = Tensor.vector("1", "2").const
    val b = "3".const

    a.concat(b).eval should be(Tensor.vector("13", "23"))
  }

  "length" should "return length of string elements" in {
    Tensor.vector("a", "aa", "aaa").const.length.eval should be(Tensor.vector(1, 2, 3))
  }

  "join" should "concatenate multiple tensors with separator" in {
    val a = Tensor.vector("1", "2").const
    val b = Tensor.vector("3", "4").const
    val c = Tensor.vector("5", "6").const

    join(", ", a, b, c).eval should be(Tensor.vector("1, 3, 5", "2, 4, 6"))
  }

//  "split" should "split string values with given separator" in {
//    Tensor.vector("a, b, c", "d, e").const.split(", ".const).eval should be(Tensor.vector("a", "b", "c", "d", "e"))
//  }
}
