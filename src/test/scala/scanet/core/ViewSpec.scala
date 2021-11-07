package scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scanet.core.Slice.syntax._

class ViewSpec extends AnyFlatSpec with Matchers {

  "projection" should "be filled from other at left side" in {
    Projection(0 until 5).alignLeft(2, 1.build) should
    be(Projection(1, 0 until 5))
  }

  it should "be filled from other at right side" in {
    Projection(0 until 5).alignRight(2, 1.build) should
    be(Projection(0 until 5, 1))
  }

  it should "be pruned when left side contains index slice" in {
    Projection(1, 0 until 5).shapePruned should be(Shape(5))
  }

  "view" should "produce right projection when same size" in {
    View(Shape(5, 5, 5), Projection(1, 2 until 4, ::)).shape should be(Shape(2, 5))
  }

  it should "produce right projection when smaller size" in {
    View(Shape(5, 5, 5), Projection(1)).shape should be(Shape(5, 5))
  }

  it should "produce right projection when index" in {
    View(Shape(5, 5, 5), Projection(1, 1, 1)).shape should be(Shape())
  }

  it should "produce right projection when unbound right" in {
    View(Shape(5, 5), Projection(::, 3 until -1)).shape should be(Shape(5, 2))
  }

  it should "fail when projection has higher rank" in {
    an[IllegalArgumentException] should be thrownBy
    View(Shape(5, 5), Projection(1, 1, 1))
  }

  it should "fail when projection is out of bound" in {
    an[IllegalArgumentException] should be thrownBy
    View(Shape(5, 5), Projection(6))
  }

  it should "be created from shape with unbound default projection" in {
    View(Shape(5, 5)).projection should be(Projection(0 until 5, 0 until 5))
  }

  it should "should be projected right first time" in {
    val view = View(Shape(5, 5, 5)) narrow Projection(1, 2 until 4, ::)
    view.shape should be(Shape(2, 5))
  }

  it should "should be projected right second time" in {
    val view = View(Shape(5, 5, 5)) narrow Projection(1, 2 until 4) narrow Projection(0 until 2)
    view.shape should be(Shape(2, 5))
  }

  it should "should fail to narrow when projection rank is greater" in {
    an[IllegalArgumentException] should be thrownBy
    View(Shape(5, 5)).narrow(Projection(1, 2 until 4, ::))
  }

  it should "should fail to narrow when projection is out of bound" in {
    an[IllegalArgumentException] should be thrownBy
    View(Shape(5, 5)).narrow(Projection(10, 10))
  }

  it should "produce positions when taking :: projection" in {
    val view = View(Shape(3, 3)) narrow Projection(::, ::)
    view.positions should be(Array(0, 1, 2, 3, 4, 5, 6, 7, 8))
  }

  it should "produce positions when taking first row" in {
    val view = View(Shape(5, 5)) narrow Projection(0, ::)
    view.positions should be(Array(0, 1, 2, 3, 4))
  }

  it should "produce positions when taking second row" in {
    val view = View(Shape(5, 5)) narrow Projection(1, ::)
    view.positions should be(Array(5, 6, 7, 8, 9))
  }

  it should "produce position when taking index" in {
    val view = View(Shape(5, 5)) narrow Projection(1, 3)
    view.positions should be(Array(8))
  }

  it should "produce positions when taking nested projection" in {
    val view = View(Shape(5, 5, 5)) narrow Projection(1, 2 until 4, ::)
    view.positions should be(Array(35, 36, 37, 38, 39, 40, 41, 42, 43, 44))
  }

  "shape" should "be broadcastable when shapes are the same" in {
    Shape(1, 2) broadcastableBy Shape(1, 2) should be(true)
  }

  it should "be broadcastable when given shape ends with second shape" in {
    Shape(2, 3, 4) broadcastableBy Shape(3, 4) should be(true)
  }

  it should "be broadcastable when second shape has ones" in {
    Shape(2, 3, 4) broadcastableBy Shape(2, 1, 4) should be(true)
  }

  it should "not be broadcastable when no broadcasting rules is found" in {
    Shape(2, 3, 4) broadcastableBy Shape(2, 2, 4) should be(false)
  }

  it should "support permutation" in {
    Shape(2, 3, 4).permute(2, 1, 0) should be(Shape(4, 3, 2))
  }

  it should "support selecting" in {
    Shape(2, 3, 4).select(1, 2) should be(Shape(3, 4))
  }

  it should "support removing" in {
    Shape(2, 3, 4).remove(1, 2) should be(Shape(2))
  }

  it should "support updating all" in {
    Shape(2, 3, 4).updateAll(1)(0, 2) should be(Shape(1, 3, 1))
  }

  "minus" should "leave higher dimensions from first shape if it is bigger" in {
    Shape(2, 3, 4) - Shape(3, 4) should be(Shape(2))
  }

  it should "return empty shape if both are equal" in {
    Shape(2, 3, 4) - Shape(2, 3, 4) should be(Shape())
  }

  it should "return empty shape if other dimension is bigger" in {
    Shape(2, 3, 4) - Shape(1, 2, 3, 4) should be(Shape())
  }

  it should "same shape if other dimension is empty" in {
    Shape(2, 3, 4) - Shape() should be(Shape(2, 3, 4))
  }

  it should "fail if shapes are incompatible" in {
    the[IllegalArgumentException] thrownBy {
      Shape(2, 3, 4) - Shape(2, 5, 4)
    } should have message "requirement failed: cannot (2, 3, 4) - (2, 5, 4)"
  }

  "shape (2, 5)" should "not be broadcastable by shape (2) or wise versa" in {
    Shape(2, 5) broadcastableAny Shape(2) should be(false)
  }

  it should "return dimension index with size one" in {
    Shape(2, 3, 4) broadcastableAxis Shape(2, 3, 1) should be(Seq(2))
  }

  it should "return empty shape if both are equal" in {
    Shape(2, 3, 4) broadcastableAxis Shape(2, 3, 4) should be(Seq())
  }

  it should "return empty shape if other dimension is bigger" in {
    Shape(2, 3, 4) broadcastableAxis Shape(1, 2, 3, 4) should be(Seq())
  }

  it should "same shape if other dimension is empty" in {
    Shape(2, 3, 4) broadcastableAxis Shape() should be(Seq(0, 1, 2))
  }

  it should "fail if shapes are incompatible" in {
    the[IllegalArgumentException] thrownBy {
      Shape(2, 3, 4) broadcastableAxis Shape(2, 5, 4)
    } should have message "requirement failed: cannot find broadcastable axis for (2, 3, 4) and (2, 5, 4)"
  }

  "broadcastable axis" should "leave higher dimensions from first shape if it is bigger" in {
    Shape(2, 3, 4) broadcastableAxis Shape(3, 4) should be(Seq(0))
  }

  "insert" should "place new dimension between existing and shift all right dimensions" in {
    Shape(2, 3, 4).insert(1, 5) should be(Shape(2, 5, 3, 4))
    Shape(2, 3, 4).insert(3, 5) should be(Shape(2, 3, 4, 5))
  }

  it should "fail when insert introduces a gap" in {
    the[IllegalArgumentException] thrownBy {
      Shape(2, 3, 4).insert(4, 5)
    } should have message "requirement failed: couldn't insert dimension 4 cause rank is 3"
  }
}
