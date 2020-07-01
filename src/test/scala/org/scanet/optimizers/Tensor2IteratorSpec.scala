package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

class Tensor2IteratorSpec extends AnyFlatSpec with CustomMatchers {

  "tensor2 iterator" should "consume entire dataset if batch is bigger than dataset" in {
    val it = Tensor2Iterator(Iterator(
      Array(1, 2, 3),
      Array(4, 5, 6)
    ), 2)
    it.hasNext should be(true)
    val x = Tensor.matrix(Array(1, 2), Array(4, 5))
    val y = Tensor.matrix(Array(3), Array(6))
    it.next() should be(x, y)
    it.hasNext should be(false)
  }

  it should "make multiple fetches if batch smaller than dataset" in {
    val it = Tensor2Iterator(Iterator(
      Array(1, 2, 3),
      Array(4, 5, 6),
      Array(7, 8, 9),
      Array(10, 11, 12)
    ), 2)
    it.hasNext should be(true)
    val x1 = Tensor.matrix(Array(1, 2), Array(4, 5))
    val y1 = Tensor.matrix(Array(3), Array(6))
    it.next() should be(x1, y1)
    it.hasNext should be(true)
    val x2 = Tensor.matrix(Array(7, 8), Array(10, 11))
    val y2 = Tensor.matrix(Array(9), Array(12))
    it.next() should be(x2, y2)
    it.hasNext should be(false)
  }

  it should "pad the batch with zeros if batch is not full and padding is enabled" in {
    val it = Tensor2Iterator(Iterator(
      Array(1, 2, 3),
      Array(4, 5, 6)
    ), 3)
    it.hasNext should be(true)
    val x = Tensor.matrix(Array(1, 2), Array(4, 5), Array(0, 0))
    val y = Tensor.matrix(Array(3), Array(6), Array(0))
    it.next() should be(x, y)
    it.hasNext should be(false)
  }

  it should "trim batch if batch is not full and padding is disabled" in {
    val it = Tensor2Iterator(Iterator(
      Array(1, 2, 3),
      Array(4, 5, 6)
    ), 3, withPadding = false)
    it.hasNext should be(true)
    val x = Tensor.matrix(Array(1, 2), Array(4, 5))
    val y = Tensor.matrix(Array(3), Array(6))
    it.next() should be(x, y)
    it.hasNext should be(false)
  }
}