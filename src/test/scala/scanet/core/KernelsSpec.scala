package scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.core.Slice.syntax.::
import scanet.syntax._

import scala.collection.immutable.Seq

class KernelsSpec extends AnyFlatSpec with Matchers {

  "scalar const" should "have toString which holds a scalar value" in {
    5.0f.const.toString should be("Const(5.0)[Float]:()")
  }

  "small vector const" should "have toString which holds all elements" in {
    Tensor.vector(1, 2).const.toString should be("Const(1, 2)[Int]:(2)")
  }

  "placeholder" should "have toString which holds an address of output object" in {
    val pl = placeholder[Int]()
    pl.toString should be(s"Placeholder(#${pl.ref})[Int]:()")
  }

  "it" should "not be equal to any other placeholder" in {
    placeholder[Int]() should not be placeholder[Int]()
  }

  "composite output" should "have toString which includes operators chain" in {
    5.0f.const.reshape(1).toString should
    be("Reshape(Const(5.0)[Float]:(),new_shape:Const(1)[Int]:(1))[Float]:(1)")
  }

  "const" should "be evaluated" in {
    5.0f.const.eval should be(Tensor.scalar(5.0f))
  }

  "product of 2 expr" should "be evaluated" in {
    (1.const, 2.const).eval should be((Tensor.scalar(1), Tensor.scalar(2)))
  }

  "product of 3 expr" should "be evaluated" in {
    (1.const, 2.const, 3.const).eval should be(
      (Tensor.scalar(1), Tensor.scalar(2), Tensor.scalar(3)))
  }

  "seq of expr" should "be evaluated" in {
    Seq(1.const, 2.const).eval should be(Seq(Tensor.scalar(1), Tensor.scalar(2)))
  }

  "seq of tuple expr" should "be evaluated" in {
    Seq((1.const, 2.const), (3.const, 4.const)).eval should be(
      Seq((Tensor.scalar(1), Tensor.scalar(2)), (Tensor.scalar(3), Tensor.scalar(4))))
  }

  "seq of seq expr" should "be evaluated" in {
    Seq(Seq(1.const, 2.const), Seq(3.const, 4.const, 5.const)).eval should be(
      Seq(
        Seq(Tensor.scalar(1), Tensor.scalar(2)),
        Seq(Tensor.scalar(3), Tensor.scalar(4), Tensor.scalar(5))))
  }

  "map of expr" should "be evaluated" in {
    Map("a" -> 1.const, "b" -> 2.const).eval should be(
      Map("a" -> Tensor.scalar(1), "b" -> Tensor.scalar(2)))
  }

  "map of tuple expr" should "be evaluated" in {
    Map("a" -> (1.const, 2.const), "b" -> (3.const, 4.const)).eval should be(
      Map("a" -> (Tensor.scalar(1), Tensor.scalar(2)), "b" -> (Tensor.scalar(3), Tensor.scalar(4))))
  }

  "map of map expr" should "be evaluated" in {
    val in = Map(
      1 -> Map("a" -> 1.const, "b" -> 2.const),
      2 -> Map("a" -> 3.const, "b" -> 4.const, "c" -> 5.const))
    val out = Map(
      1 -> Map("a" -> Tensor.scalar(1), "b" -> Tensor.scalar(2)),
      2 -> Map("a" -> Tensor.scalar(3), "b" -> Tensor.scalar(4), "c" -> Tensor.scalar(5)))
    in.eval should be(out)
  }

  "params of expr" should "be evaluated" in {
    Params(Path("a") -> 1.const, Path("b") -> 2.const).eval should be(
      Params(Path("a") -> Tensor.scalar(1), Path("b") -> Tensor.scalar(2)))
  }

  "reshape" should "transform vector into matrix" in {
    Tensor.vector(1, 2, 3, 4).const.reshape(2, 2).eval should be(
      Tensor.matrix(Array(1, 2), Array(3, 4)))
  }

  it should "transform matrix into vector" in {
    Tensor.matrix(Array(1, 2), Array(3, 4)).const.reshape(4).eval should be(
      Tensor.vector(1, 2, 3, 4))
  }

  it should "transform matrix into another matrix" in {
    Tensor.matrix(Array(1, 2), Array(3, 4)).const.reshape(4).eval should be(
      Tensor.vector(1, 2, 3, 4))
  }

  it should "fail when power does not match" in {
    the[IllegalArgumentException] thrownBy {
      Tensor.range(0 until 7).const.reshape(4, 4)
    } should have message "requirement failed: shape (7) cannot be reshaped into (4, 4)"
  }

  it should "support gradient" in {
    val x = Tensor.vector(1, 2, 3, 4).const
    x.reshape(2, 2).sum.grad(x).returns[Float].eval should be(Tensor.vector(1, 1, 1, 1))
  }

  "squeeze" should "convert matrix into a vector when first dimension is 1" in {
    Tensor.matrix(Array(1, 2, 3)).const.squeeze.eval should be(Tensor.vector(1, 2, 3))
  }

  it should "support gradient" in {
    val x = Tensor.matrix(Array(1, 2, 3)).const
    x.squeeze.sum.grad(x).returns[Float].eval should be(Tensor.matrix(Array(1, 1, 1)))
  }

  "cast" should "convert vector of Floats into Ints" in {
    Tensor.vector(1.2f, 2.2f, 3.3f).const.cast[Int].eval should be(Tensor.vector(1, 2, 3))
  }

  "cast" should "support gradient" in {
    val a = Tensor.vector(1, 2, 3).const
    val x = Tensor.vector(5.2f, 10.2f, 15.3f).const
    ((x.cast[Int] + a).sum grad x).returns[Float].eval should be(Tensor.vector(1.0f, 1.0f, 1.0f))
  }

  "depends" should "calculate one node after another" in {
    val a = 1.const
    val b = 2.const
    val c = 3.const
    val z = (a + b) << (b + c)
    z.eval should be(Tensor.scalar(3))
  }

  "when" should "calculate output based on true condition" in {
    val a = 1.const
    val b = 0.const
    val c = 2.const

    val ternary = when(a gt b) thenDo (a plus c) elseDo (a minus c)
    ternary.eval should be(Tensor.scalar(3))
  }

  it should "calculate output based on false condition" in {
    val a = -1.const
    val b = 0.const
    val c = 2.const

    val ternary = when(a gt b) thenDo (a plus c) elseDo (a minus c)
    ternary.eval should be(Tensor.scalar(-3))
  }

  "slice" should "index a matrix to a row" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.slice(0).eval should be(Tensor.vector(1, 2, 3))
  }

  it should "index a matrix to an element" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.slice(0, 0).eval should be(Tensor.scalar(1))
  }

  it should "slice a matrix" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    x.slice(::, 0 until 2).eval should be(Tensor.matrix(Array(1, 2), Array(4, 5)))
  }

  it should "fail when out of bounds" in {
    the[IllegalArgumentException] thrownBy {
      Tensor.eye[Int](3).const.slice(1, 1, 1)
    } should have message "requirement failed: projection (1, 1, 1) is out of bound, should fit shape (3, 3)"
  }

  it should "have valid gradient" in {
    val x = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    val y = x.slice(0).sum
    val grad = y.grad(x).returns[Float].eval
    grad shouldBe Tensor.matrix(
      Array(1.0, 1.0, 1.0),
      Array(0.0, 0.0, 0.0))
  }

  "pad" should "add padding to the tensor" in {
    val x = Tensor.matrix(Array(1, 2), Array(3, 4)).const
    val y = x.pad(Seq((1, 1), (1, 0)), 0)
    y.eval shouldBe Tensor.matrix(
      Array(0, 0, 0),
      Array(0, 1, 2),
      Array(0, 3, 4),
      Array(0, 0, 0))
  }

  "join" should "concat 2 vectors" in {
    val a = Tensor.vector(1, 2).const
    val b = Tensor.vector(3, 4).const
    (a join b).eval should be(Tensor.vector(1, 2, 3, 4))
  }

  it should "concat 2 matrices rows" in {
    val a = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    val b = Tensor.matrix(Array(7, 8, 9), Array(10, 11, 12)).const
    val ab = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6), Array(7, 8, 9), Array(10, 11, 12))
    (a join b).eval should be(ab)
  }

  it should "concat 2 matrices columns" in {
    val a = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    val b = Tensor.matrix(Array(7, 8, 9), Array(10, 11, 12)).const
    val ab = Tensor.matrix(Array(1, 2, 3, 7, 8, 9), Array(4, 5, 6, 10, 11, 12))
    a.joinAlong(b, 1).eval should be(ab)
  }

  it should "fail when dimensions do not match" in {
    the[IllegalArgumentException] thrownBy {
      val a = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
      val b = Tensor.matrix(Array(7, 8), Array(9, 10)).const
      a join b
    } should have message "requirement failed: " +
    "all inputs should have same dimensions except the axis, but was (2, 3), (2, 2)"
  }

  it should "have valid gradient" in {
    val a = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    val b = Tensor.matrix(Array(7, 8, 9)).const
    val f = ((a join b) * 2.const).sum
    f.grad(a).returns[Float].eval should be(Tensor.fill(2, 3)(2))
    f.grad(b).returns[Float].eval should be(Tensor.fill(1, 3)(2))
  }

  "zip" should "pack 2 vectors into a matrix" in {
    val first = Tensor.vector(1, 2, 3).const
    val second = Tensor.vector(4, 5, 6).const
    val expected = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6))
    (first zip second).eval should be(expected)
  }

  "unzip" should "unpack 2 vectors from a matrix" in {
    val matrix = Tensor.matrix(Array(1, 2, 3), Array(4, 5, 6)).const
    matrix.unzip.eval should be((Tensor.vector(1, 2, 3), Tensor.vector(4, 5, 6)))
  }

  "fill" should "fill a tensor with a given value" in {
    fill(2, 2)(1).eval should be(Tensor.matrix(Array(1, 1), Array(1, 1)))
  }

  it should "have correct gradient" in {
    val x = 5.const
    val f = (fillOutput(2)(x) * Tensor.vector(4, 5).const).sum
    f.grad(x).returns[Float].eval should be(Tensor.scalar(9f))
  }
}
