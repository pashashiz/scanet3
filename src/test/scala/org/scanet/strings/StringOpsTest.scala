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
    Tensor.vector("1", "2").const.concat("3".const).eval should be(Tensor.vector("13", "23"))
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

  "toNumber" should "parse string values" in {
    Tensor.vector("1.1", "2.2", "3.3").const.toNumber[Float].eval should be(Tensor.vector(1.1f, 2.2f, 3.3f))
  }

  "substring" should "" in {
    val strings = Tensor.matrix(
      Array("ten", "eleven", "twelve"),
      Array("thirteen", "fourteen", "fifteen'"),
      Array("sixteen", "seventeen", "eighteen"))
    val positions = Tensor.matrix(Array(1, 2, 3), Array(1, 2, 3), Array(1, 2, 3))
    val lengths = Tensor.matrix(Array(2, 3, 4), Array(4, 3, 2), Array(5, 5, 5))

    val expected = Tensor.matrix(
      Array("en", "eve", "lve"),
      Array("hirt", "urt", "te"),
      Array("ixtee", "vente", "hteen"))

    strings.const.substring(positions.const, lengths.const).eval should be(expected)
  }

  //  "split" should "split string values with given separator" in {
  //    Tensor.vector("a, b, c", "d, e").const.split(", ".const).eval should be(Tensor.vector("a", "b", "c", "d", "e"))
  //  }
}
