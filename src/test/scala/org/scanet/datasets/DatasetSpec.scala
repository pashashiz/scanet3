package org.scanet.datasets

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.{Shape, Tensor}
import org.scanet.math.syntax._

class DatasetSpec extends AnyFlatSpec with Matchers {

  "tensor dataset" should "produce next batch" in {
    val it = TensorDataset(Tensor.vector(1, 2, 3)).iterator
    it.hasNext should be(true)
    it.next(2) should be(Tensor.vector(1, 2))
    it.hasNext should be(true)
    it.next(2) should be(Tensor.vector(2, 3))
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

  it should "produce correct shapes" in {
    val it = TensorDataset(Tensor.vector(1, 2, 3)).iterator
    it.shape should be(Shape())
  }

  "CSV dataset" should "read records" in {
    val ds = CSVDataset("linear_function_1.scv")
    val it = ds.iterator
    it.size should be(97)
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
      Array(5.8707f, 7.2029f),
      Array(5.3054f, 1.9869f),
      Array(8.2934f, 0.14454f),
      Array(13.394f, 9.0551f),
      Array(5.4369f, 0.61705f)
    ))
  }

  it should "produce correct shapes" in {
    val it = CSVDataset("linear_function_1.scv").iterator
    it.shape should be(Shape(2))
  }
}
