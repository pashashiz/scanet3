package scanet.core

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Slice.syntax._

class ViewSpec extends AnyWordSpec with Matchers {

  "projection" should {

    "be filled from other at left side" in {
      Projection(0 until 5).alignLeft(2, 1.build) should
      be(Projection(1, 0 until 5))
    }

    "be filled from other at right side" in {
      Projection(0 until 5).alignRight(2, 1.build) should
      be(Projection(0 until 5, 1))
    }

    "be pruned when left side contains index slice" in {
      Projection(1, 0 until 5).shapePruned should be(Shape(5))
    }
  }

  "view" should {

    "produce right projection when same size" in {
      View(Shape(5, 5, 5), Projection(1, 2 until 4, ::)).shape should be(Shape(2, 5))
    }

    "produce right projection when smaller size" in {
      View(Shape(5, 5, 5), Projection(1)).shape should be(Shape(5, 5))
    }

    "produce right projection when index" in {
      View(Shape(5, 5, 5), Projection(1, 1, 1)).shape should be(Shape())
    }

    "produce right projection when unbound right" in {
      View(Shape(5, 5), Projection(::, 3 until -1)).shape should be(Shape(5, 2))
    }

    "fail when projection has higher rank" in {
      an[IllegalArgumentException] should be thrownBy
      View(Shape(5, 5), Projection(1, 1, 1))
    }

    "fail when projection is out of bound" in {
      an[IllegalArgumentException] should be thrownBy
      View(Shape(5, 5), Projection(6))
    }

    "be created from shape with unbound default projection" in {
      View(Shape(5, 5)).projection should be(Projection(0 until 5, 0 until 5))
    }

    "should be projected right first time" in {
      val view = View(Shape(5, 5, 5)) narrow Projection(1, 2 until 4, ::)
      view.shape should be(Shape(2, 5))
    }

    "should be projected right second time" in {
      val view = View(Shape(5, 5, 5)) narrow Projection(1, 2 until 4) narrow Projection(0 until 2)
      view.shape should be(Shape(2, 5))
    }

    "should fail to narrow when projection rank is greater" in {
      an[IllegalArgumentException] should be thrownBy
      View(Shape(5, 5)).narrow(Projection(1, 2 until 4, ::))
    }

    "should fail to narrow when projection is out of bound" in {
      an[IllegalArgumentException] should be thrownBy
      View(Shape(5, 5)).narrow(Projection(10, 10))
    }

    "produce positions when taking :: projection" in {
      val view = View(Shape(3, 3)) narrow Projection(::, ::)
      view.positions should be(Array(0, 1, 2, 3, 4, 5, 6, 7, 8))
    }

    "produce positions when taking first row" in {
      val view = View(Shape(5, 5)) narrow Projection(0, ::)
      view.positions should be(Array(0, 1, 2, 3, 4))
    }

    "produce positions when taking second row" in {
      val view = View(Shape(5, 5)) narrow Projection(1, ::)
      view.positions should be(Array(5, 6, 7, 8, 9))
    }

    "produce position when taking index" in {
      val view = View(Shape(5, 5)) narrow Projection(1, 3)
      view.positions should be(Array(8))
    }

    "produce positions when taking nested projection" in {
      val view = View(Shape(5, 5, 5)) narrow Projection(1, 2 until 4, ::)
      view.positions should be(Array(35, 36, 37, 38, 39, 40, 41, 42, 43, 44))
    }
  }

  "shape" should {

    "be broadcastable when shapes are the same" in {
      Shape(1, 2) broadcastableBy Shape(1, 2) should be(true)
    }

    "be broadcastable when given shape ends with second shape" in {
      Shape(2, 3, 4) broadcastableBy Shape(3, 4) should be(true)
    }

    "be broadcastable when second shape has ones" in {
      Shape(2, 3, 4) broadcastableBy Shape(2, 1, 4) should be(true)
    }

    "not be broadcastable when no broadcasting rules is found" in {
      Shape(2, 3, 4) broadcastableBy Shape(2, 2, 4) should be(false)
    }

    "support permutation" in {
      Shape(2, 3, 4).permute(2, 1, 0) should be(Shape(4, 3, 2))
    }

    "support selecting" in {
      Shape(2, 3, 4).select(1, 2) should be(Shape(3, 4))
    }

    "support selecting of last dimension" in {
      Shape(2, 3, 4).select(1, -1) should be(Shape(3, 4))
    }

    "support removing" in {
      Shape(2, 3, 4).remove(1, 2) should be(Shape(2))
    }

    "support updating all" in {
      Shape(2, 3, 4).updateAll(1)(0, 2) should be(Shape(1, 3, 1))
    }

    "support updating all except" in {
      Shape(2, 3, 4).updateAllExcept(1)(0, 2) should be(Shape(2, 1, 4))
    }

    "have minus operation" which {

      "should leave higher dimensions from first shape if it is bigger" in {
        Shape(2, 3, 4) - Shape(3, 4) should be(Shape(2))
      }

      "should return empty shape if both are equal" in {
        Shape(2, 3, 4) - Shape(2, 3, 4) should be(Shape())
      }

      "should return empty shape if other dimension is bigger" in {
        Shape(2, 3, 4) - Shape(1, 2, 3, 4) should be(Shape())
      }

      "should same shape if other dimension is empty" in {
        Shape(2, 3, 4) - Shape() should be(Shape(2, 3, 4))
      }

      "should fail if shapes are incompatible" in {
        the[IllegalArgumentException] thrownBy {
          Shape(2, 3, 4) - Shape(2, 5, 4)
        } should have message "requirement failed: cannot (2, 3, 4) - (2, 5, 4)"
      }
    }

    "not be broadcastable by shape (2) if it is (2, 5) or wise versa" in {
      Shape(2, 5) broadcastableAny Shape(2) should be(false)
    }

    "have broadcastable axis operation" which {

      "should leave higher dimensions from first shape if it is bigger" in {
        Shape(2, 3, 4) broadcastableAxes Shape(3, 4) should be(Seq(0))
      }

      "should return dimension index with size one" in {
        Shape(2, 3, 4) broadcastableAxes Shape(2, 3, 1) should be(Seq(2))
      }

      "should return empty shape if both are equal" in {
        Shape(2, 3, 4) broadcastableAxes Shape(2, 3, 4) should be(Seq())
      }

      "should return empty shape if other dimension is bigger" in {
        Shape(2, 3, 4) broadcastableAxes Shape(1, 2, 3, 4) should be(Seq())
      }

      "should same shape if other dimension is empty" in {
        Shape(2, 3, 4) broadcastableAxes Shape() should be(Seq(0, 1, 2))
      }

      "should fail if shapes are incompatible" in {
        the[IllegalArgumentException] thrownBy {
          Shape(2, 3, 4) broadcastableAxes Shape(2, 5, 4)
        } should have message "requirement failed: cannot find broadcastable axis for (2, 3, 4) and (2, 5, 4)"
      }
    }

    "have insert operation" which {

      "should place new dimension between existing and shift all right dimensions" in {
        Shape(2, 3, 4).insert(1, 5) should be(Shape(2, 5, 3, 4))
        Shape(2, 3, 4).insert(3, 5) should be(Shape(2, 3, 4, 5))
      }

      "should fail when insert introduces a gap" in {
        the[IllegalArgumentException] thrownBy {
          Shape(2, 3, 4).insert(4, 5)
        } should have message "requirement failed: couldn't insert dimension 4 cause rank is 3"
      }
    }

    "shift dimensions to the right preserving all existing dimensions and fill left new dimensions with a given value" in {
      Shape(2, 3, 4) >>> (2, 1) shouldBe Shape(1, 1, 2, 3, 4)
      Shape(2, 3, 4) >>> 2 shouldBe Shape(1, 1, 2, 3, 4)
    }

    "shift dimensions to the left preserving all existing dimensions  and fill left new dimensions with a given value" in {
      Shape(2, 3, 4) <<< (2, 1) shouldBe Shape(2, 3, 4, 1, 1)
      Shape(2, 3, 4) <<< 2 shouldBe Shape(2, 3, 4, 1, 1)
    }

    "shift dimensions to the right dropping dimensions which are out of window" in {
      Shape(2, 3, 4) << 2 shouldBe Shape(4)
    }

    "shift dimensions to the left dropping dimensions which are out of window" in {
      Shape(2, 3, 4) >> 2 shouldBe Shape(2)
    }

    "found max dimension of both shapes" in {
      Shape(2, 3, 4).maxDims(Shape(5, 2, 2)) shouldBe Shape(5, 3, 4)
      Shape(2, 3, 4).maxDims(Shape(2, 2)) shouldBe Shape(2, 3, 4)
      Shape(2, 3, 4).maxDims(Shape(2, 5)) shouldBe Shape(2, 3, 5)
      Shape(2, 3, 4).maxDims(Shape(6)) shouldBe Shape(2, 3, 6)
    }
  }
}
