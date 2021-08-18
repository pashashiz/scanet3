package org.scanet.strings

import org.scalatest.Ignore
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.syntax._

import scala.reflect.io.Path._

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

  "substring" should "calculate substrings by given pos and len" in {
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

  "vector" should "be converted into String" in {
    Tensor.vector(1, 2, 3).const.asString.eval should be(Tensor.vector("1", "2", "3"))
  }

  it should "format vector into string scalar" in {
    Tensor.vector(1, 2, 3).const.format.eval should be(Tensor.scalar("[1 2 3]"))
  }

  it should "format matrix summary" in {
    val matrix = Tensor.matrix(
      Array(1, 2),
      Array(3, 4),
      Array(5, 6),
      Array(7, 8),
      Array(9, 10),
      Array(11, 12),
      Array(13, 14),
      Array(15, 16)
    )

    val formatted = matrix.const.format("matrix: {}").eval
    formatted should be(Tensor.scalar("matrix: [[1 2]\n [3 4]\n [5 6]\n ...\n [11 12]\n [13 14]\n [15 16]]"))
  }

  it should "print and return the same vector" in {
    Tensor.vector(1, 2).const.print.eval should be(Tensor.vector(1, 2))
  }

  it should "print graph vertices" in {
    val a = Tensor.matrix(
      Array(1, 2),
      Array(1, 2))
    val b = Tensor.vector(1, 2)
    val c = Tensor.matrix(
      Array(2, 4),
      Array(2, 4))

    val file = "test_" + System.currentTimeMillis() + ".txt"
    val out = ToFile(file)
    (a.const.print(out) plus b.const).print(out).eval should be(c)

    file.toFile.exists should be(true)
    val source = scala.io.Source.fromFile(file)
    source.mkString should be("[[1 2]\n [1 2]]\n[[2 4]\n [2 4]]\n")
    source.close()
    file.toFile.delete()
  }

  it should "print multiple tensors" in {
    val a = 1.const
    val b = 2.const
    val c = (a plus b) << print("a + b = {} + {}", a, b)

    c.eval should be(Tensor.scalar(3))
  }

  "assert" should "verify current output" in {
    val a = Tensor.vector(1, 2).const
    val b = Tensor.vector(3, 4).const
    val c = Tensor.vector(4, 6).const

    (a plus b).assert(_ === c).eval should be(Tensor.vector(4, 6))
  }

  "assert" should "throw when current condition is false" in {
    the [IllegalArgumentException] thrownBy {
      val a = Tensor.vector(1, 2).const
      val b = Tensor.vector(3, 4).const
      val c = Tensor.vector(5, 6).const

      (a plus b).assert(_ === c).eval
    } should have message "assertion failed: [value: [4 6]]\n\t [[{{node Assert_0}}]]"
  }

  "assert that" should "verify dependent condition" in {
    val a = 1.const
    val b = 2.const
    val c = (a plus b) << assertThat((a gt 0.const) and (b gt 0.const), "{} {}", a, b)

    c.eval should be(Tensor.scalar(3))
  }

  "assert that" should "should fail when dependent condition is false" in {
    the [IllegalArgumentException] thrownBy {
      val a = Tensor.vector(1, 2).const
      val b = 0.const
      val c = (a div b) << assertThat(b gt 0.const, "b was {}, should be > 0", b)

      c.eval should be(Tensor.scalar(3))
    } should have message "assertion failed: [b was 0, should be > 0]\n\t [[{{node Assert_0}}]]"
  }
}
