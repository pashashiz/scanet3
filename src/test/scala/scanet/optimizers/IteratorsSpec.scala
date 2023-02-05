package scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.{Shape, Tensor}
import scanet.math.syntax._
import scanet.optimizers.Iterators.{PadZeros, Partial}
import scanet.test.CustomMatchers

class IteratorsSpec extends AnyFlatSpec with CustomMatchers {

  "tensor iterator" should "consume entire dataset if batch is bigger than dataset" in {
    val it =
      TensorIterator(
        rows = Iterator(
          Record(Array(1, 2), Array(3)),
          Record(Array(4, 5), Array(6))),
        shapes = (Shape(2), Shape(1)),
        batch = 2)
    it.hasNext should be(true)
    val x = Tensor.matrix(Array(1, 2), Array(4, 5))
    val y = Tensor.matrix(Array(3), Array(6))
    it.next() should be((x, y))
    it.hasNext should be(false)
  }

  it should "make multiple fetches if batch smaller than dataset" in {
    val it = TensorIterator(
      rows = Iterator(
        Record(Array(1, 2), Array(3)),
        Record(Array(4, 5), Array(6)),
        Record(Array(7, 8), Array(9)),
        Record(Array(10, 11), Array(12))),
      shapes = (Shape(2), Shape(1)),
      batch = 2)
    it.hasNext should be(true)
    val x1 = Tensor.matrix(Array(1, 2), Array(4, 5))
    val y1 = Tensor.matrix(Array(3), Array(6))
    it.next() should be((x1, y1))
    it.hasNext should be(true)
    val x2 = Tensor.matrix(Array(7, 8), Array(10, 11))
    val y2 = Tensor.matrix(Array(9), Array(12))
    it.next() should be((x2, y2))
    it.hasNext should be(false)
  }

  it should "pad the batch with zeros if batch is not full and padding is enabled" in {
    val it = TensorIterator(
      rows = Iterator(
        Record(Array(1, 2), Array(3)),
        Record(Array(4, 5), Array(6))),
      shapes = (Shape(2), Shape(1)),
      batch = 3,
      remaining = PadZeros)
    it.hasNext should be(true)
    val x = Tensor.matrix(Array(1, 2), Array(4, 5), Array(0, 0))
    val y = Tensor.matrix(Array(3), Array(6), Array(0))
    it.next() should be((x, y))
    it.hasNext should be(false)
  }

  it should "trim batch if batch is not full and padding is disabled" in {
    val it = TensorIterator(
      rows = Iterator(
        Record(Array(1, 2), Array(3)),
        Record(Array(4, 5), Array(6))),
      shapes = (Shape(2), Shape(1)),
      batch = 3,
      remaining = Partial)
    it.hasNext should be(true)
    val x = Tensor.matrix(Array(1, 2), Array(4, 5))
    val y = Tensor.matrix(Array(3), Array(6))
    it.next() should be((x, y))
    it.hasNext should be(false)
  }
}
