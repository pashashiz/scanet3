package org.scanet.datasets

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._

class DatasetSpec extends AnyFlatSpec with Matchers {

  "tensor dataset" should "produce next batch" in {
    val it = TensorDataset(Tensor.vector(1, 2, 3)).iterator
    it.hasNext should be(true)
    it.next(2) should be(Tensor.vector(1, 2))
    it.hasNext should be(true)
    it.next(2) should be(Tensor.vector(3, 0))
  }

  it should "fail when requested batch is out of bound" in {
    val it = TensorDataset(Tensor.vector(1, 2, 3)).iterator
    it.hasNext should be(true)
    it.next(3) should be(Tensor.vector(1, 2, 3))
    it.hasNext should be(false)
    the [IllegalArgumentException] thrownBy {
      it.next(3)
    } should have message "requirement failed: dataset has no elements left"
  }

  "CSV dataset" should "read records" in {
    val ds = CSVDataset("linear_function_1.scv")
    ds.size should be(97)
    val it = ds.iterator
    it.next(2) should be(Tensor.matrix(
      Array(6.1101f, 17.592f),
      Array(5.5277f, 9.1302f)))
  }

  it should "fill last batch with zeros" in {
    val it = CSVDataset("linear_function_1.scv").iterator
    // rewind to last batch
    for (_ <- 1 to 19) {
      it.next(5)
    }
    it.next(5) should be(Tensor.matrix(
      Array(13.394f, 9.0551f),
      Array(5.4369f, 0.61705f),
      Array(0f, 0f),
      Array(0f, 0f),
      Array(0f, 0f)
    ))
  }
}
