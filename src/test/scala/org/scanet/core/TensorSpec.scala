package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Slice.syntax.::
import org.scanet.core.TensorType.IntTag
import org.scanet.core.syntax._
import org.scanet.math.Dist.syntax._
import org.scanet.math.Generator.uniform
import org.scanet.math.Numeric.syntax._
import org.scanet.test.CustomMatchers

class TensorSpec extends AnyFlatSpec with CustomMatchers {

  "scalar" should "be allocated" in {
    Tensor.scalar(5) should
      (haveShape (Shape()) and containData (Array(5)))
  }

  "vector" should "be allocated with Int" in {
    Tensor.vector(1, 2, 3) should
      (haveShape (Shape(3)) and containData (Array(1, 2, 3)))
  }

  "vector" should "be allocated with Boolean" in {
    Tensor.vector(true, false, true) should
      (haveShape (Shape(3)) and containData (Array(true, false, true)))
  }

  "vector" should "be allocated with String" in {
    Tensor.vector("Hello", "World", "From TensorFlow") should
      (haveShape (Shape(3)) and containData (Array("Hello", "World", "From TensorFlow")))
  }

  "vector" should "be allocated with long String" in {
    Tensor.vector("", "abc" * 100, "a" * Byte.MaxValue, "a" * 16384, "abcdef" * 100) should
      (haveShape (Shape(5)) and containData (Array("", "abc" * 100, "a" * Byte.MaxValue, "a" * 16384, "abcdef" * 100)))
  }

  "matrix" should "be allocated" in {
      Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)) should
        (haveShape (Shape(2, 3)) and containData (Array(1, 2, 3, 4, 5, 6)))
  }

  "n3 tensor" should "be allocated" in {
    Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2)) should
      (haveShape (Shape(2, 2, 2)) and containData (Array(1, 2, 3, 4, 5, 6, 7, 8)))
  }

  it should "fail to be constructed when shape does not match the input" in {
    an [IllegalArgumentException] should be thrownBy
      Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 1))
  }

  "zeros tensor" should "be allocated" in {
    Tensor.zeros[Int](2, 2) should be(Tensor.matrix(Array(0, 0), Array(0, 0)))
  }

  "matrix" should "be filled with a given number" in {
    Tensor.fill(2, 2)(1) should be(Tensor.matrix(Array(1, 1), Array(1, 1)))
  }

  it should "be tabulated" in {
     Tensor.tabulate[Int](2, 2)((i, j) => (i + 1) * (j + 1)) should
       be(Tensor.matrix(Array(1, 2), Array(2, 4)))
  }

  "diagonal matrix" should "be created" in {
    Tensor.diag[Int](1, 2, 3) should be(
      Tensor.matrix(
        Array(1, 0, 0),
        Array(0, 2, 0),
        Array(0, 0, 3)))
  }

  "eye matrix" should "be created" in {
    Tensor.eye[Int](3) should be(
      Tensor.matrix(
        Array(1, 0, 0),
        Array(0, 1, 0),
        Array(0, 0, 1)))
  }

  "linspace vector" should "be created for Int" in {
    Tensor.linspace(2, 10, 5) should
      be(Tensor.vector(2, 4, 6, 8, 10))
  }

  it should "be created for Float" in {
    Tensor.linspace(2.0f, 4.0f, 5) should
      be(Tensor.vector(2.0f, 2.5f, 3.0f, 3.5f, 4.0f))
  }

  "vector" should "be created from a range" in {
    Tensor.vector(1 to 10 by 2) should
      be(Tensor.vector(1, 3, 5, 7, 9))
  }

  it should "be created from exclusive range" in {
    Tensor.range(1, 5, 2) should be(Tensor.vector(1, 3))
  }

  it should "be created from exclusive range with float" in {
    Tensor.range(1.0f, 5.0f, 2.1f) should be(Tensor.vector(1f, 3.1f))
  }

  it should "be created from inclusive range" in {
    Tensor.range(1, 5, 2, inclusive = true) should be(Tensor.vector(1, 3, 5))
  }

  it should "be created from inclusive range with float" in {
    Tensor.range(1.0f, 6.0f, 2.5f, inclusive = true) should be(Tensor.vector(1f, 3.5f, 6.0f))
  }

  "random Int tensor" should "be created with uniform distribution" in {
    Tensor.rand[Int](Shape(3), uniform(1)) should
      be(Tensor.vector(384748, -1151252339, -549383847))
  }

  "random Float tensor" should "be created with uniform distribution" in {
    Tensor.rand[Float](Shape(3), uniform(1)) should
      be(Tensor.vector(8.952618E-5f, 0.73195314f, 0.8720866f))
  }

  "random Double tensor" should "be created with uniform distribution" in {
    Tensor.rand[Double](Shape(3), uniform(1L)) should
      be(Tensor.vector(8.958178688844853E-5, 0.872086605065287, 0.7943048233411579))
  }

  "scalar" should "be indexed" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.scalar(5).slice(0)
    } should have message "requirement failed: " +
      "projection (0) has rank '1' which is greater than shape's rank '0'"
  }

  "vector" should "be indexed" in {
    Tensor.vector(0, 1, 2).slice(1) should be(Tensor.scalar(1))
  }

  it should "be sliced with closed range" in {
    Tensor.vector(0, 1, 2, 3).slice(1 until 3) should be(Tensor.vector(1, 2))
  }

  it should "be sliced with right opened range" in {
    Tensor.vector(0, 1, 2, 3).slice(1 until -1) should be(Tensor.vector(1, 2, 3))
  }

  it should "be sliced with one-element range and not be collapsed" in {
    Tensor.vector(0, 1, 2, 3).slice(1 until 2) should be(Tensor.vector(1))
  }

  it should "remain identical when sliced with unbound range" in {
    Tensor.vector(0, 1, 2, 3).slice(::) should be(Tensor.vector(0, 1, 2, 3))
  }

  it should "fail to slice when higher rank" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.vector(0, 1, 2).slice(0, 0)
    } should have message "requirement failed: " +
      "projection (0, 0) has rank '2' which is greater than shape's rank '1'"
  }

  it should "fail to slice when out of bounds" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.vector(0, 1, 2).slice(5)
    } should have message "requirement failed: " +
      "projection (5) is out of bound, should fit shape (3)"
  }

  "matrix" should "be indexed" in {
    Tensor.eye[Int](3).slice(0, 0) should be(Tensor.scalar(1))
  }

  it should "be sliced by unbound range -> closed range" in {
    val matrix = Tensor.eye[Int](3)
    matrix(::, 1 until 3) should be(Tensor.matrix(Array(0, 0), Array(1, 0), Array(0, 1)))
  }

  it should "be sliced 2 times" in {
    val matrix = Tensor.eye[Int](3)
    val vector = matrix(0)
    val slicedVector = vector(1 until -1)
    slicedVector should be(Tensor.vector(0, 0))
  }

  it should "fail to slice when out of bounds" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.eye[Int](3).slice(1, 1, 1)
    } should have message "requirement failed: " +
      "projection (1, 1, 1) has rank '3' which is greater than shape's rank '2'"
  }

  "n3 tensor" should "be indexed" in {
    val tensor = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2))
    tensor(1, 1, 1) should be(Tensor.scalar(8))
  }

  it should "be sliced" in {
    val tensor = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8), Shape(2, 2, 2))
    tensor(1, 1, ::) should be(Tensor.vector(7, 8))
  }

  "non sliced vector" should "be reshaped into matrix" in {
    Tensor.vector(0, 1, 2, 3).reshape(2, 2) should be(Tensor.matrix(Array(0, 1), Array(2, 3)))
  }

  "non sliced matrix" should "be reshaped into vector" in {
    Tensor.matrix(Array(0, 1), Array(2, 3)).reshape(4) should be(Tensor.vector(0, 1, 2, 3))
  }

  "sliced vector" should "be reshaped into matrix" in {
    val matrix = Tensor.range(0 until 7).slice(0 until 4).reshape(2, 2)
    matrix should be(Tensor.matrix(Array(0, 1), Array(2, 3)))
    matrix.view.toString should be("(7) x (:4) = (4) -> (2, 2) x (:2, :2) = (2, 2)")
  }

  "sliced reshaped into matrix vector" should "be sliced again" in {
    val vector = Tensor.range(0 until 7).slice(0 until 4).reshape(2, 2).slice(0)
    vector should be(Tensor.vector(0, 1))
    vector.view.toString should be("(7) x (:4) = (4) -> (2, 2) x (0, :2) = (2)")
  }

  "non sliced tensor" should "fail to slice when power does not match" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.range(0 until 7).reshape(4, 4)
    } should have message "requirement failed: shape (7) cannot be reshaped into (4, 4)"
  }

  "sliced tensor" should "fail to slice when power does not match" in {
    the [IllegalArgumentException] thrownBy {
      Tensor.range(0 until 7).slice(0 until 4).reshape(4, 4)
    } should have message "requirement failed: shape (4) cannot be reshaped into (4, 4)"
  }

  "scalar tensor" should "fold left" in {
    Tensor.scalar(5).foldLeft(0)(_ + _.toScalar) should be(5)
  }

  "vector tensor" should "fold left" in {
    Tensor.range(0 until 7).foldLeft(0)(_ + _.toScalar) should be(21)
  }

  "matrix tensor" should "fold left" in {
    Tensor.eye[Int](3).foldLeft(0)(_ + _.power) should be(9)
  }

  "scalar tensor" should "be shown" in {
    Tensor.scalar(5).toString should be("Tensor[Int](shape=()): 5")
  }

  "vector tensor" should "be shown" in {
    Tensor.range(0 until 7).toString should
      be("Tensor[Int](shape=(7)): [0, 1, 2, 3, 4, 5, 6]")
  }

  "matrix tensor" should "be shown" in {
    Tensor.eye[Int](5).toString should be(
      """Tensor[Int](shape=(5, 5)):
        |[
        |  [1, 0, 0, 0, 0],
        |  [0, 1, 0, 0, 0],
        |  [0, 0, 1, 0, 0]
        |]"""
       .stripMargin)
  }

  "n3 tensor" should "be shown" in {
    Tensor.rand[Int](Shape(3, 3, 3), uniform(1)).toString should be(
      """Tensor[Int](shape=(3, 3, 3)):
        |[
        |  [
        |    [384748, -1151252339, -549383847],
        |    [1612966641, -883454042, 1563994289],
        |    [1331515492, -234691648, 672332705]
        |  ],
        |  [
        |    [-2039128390, -1888584533, -294927845],
        |    [1517050556, 92416162, -1713389258],
        |    [2059776629, -1292618668, 562838985]
        |  ]
        |]"""
        .stripMargin)
  }

  "compact" should "compact sliced tensor" in {
    Tensor.range(0 until 7).slice(0 until 2).compact.hasView should be(false)
  }

  "compact" should "not compact not sliced tensor" in {
    val tensor = Tensor.range(0 until 7)
    tensor.compact should be theSameInstanceAs tensor
  }

  "Int tensor" should "be serialized to bytes array" in {
    val bytes = Array[Byte](
      5, 0, 0, 0,
      7, 0, 0, 0)
    Tensor.vector[Int](5, 7).toBytes should be(bytes)
  }

  it should "be deserialized from bytes array" in {
    val bytes = Array[Byte](
      5, 0, 0, 0,
      7, 0, 0, 0)
    Tensor.fromBytes[Int](bytes, Shape(2)) should be(Tensor.vector[Int](5, 7))
  }

  "tensor with view" should "be compacted when serialized to bytes array" in {
    val bytes = Array[Byte](
      0, 0, 0, 0,
      1, 0, 0, 0)
    Tensor.range(0 until 7).slice(0 until 2).toBytes should be(bytes)
  }

  "untyped tensor" should "be deserialized from bytes array" in {
    val bytes = Array[Byte](
      5, 0, 0, 0,
      7, 0, 0, 0)
    Tensor.fromBytesUntyped(IntTag, bytes, Shape(2)) should be(Tensor.vector[Int](5, 7))
  }
}