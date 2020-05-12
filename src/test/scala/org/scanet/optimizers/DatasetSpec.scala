package org.scanet.optimizers

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
    it.next(2) should be(Tensor.vector(3))
  }

  "tensor dataset" should "fail when requested batch is out of bound" in {
    val it = TensorDataset(Tensor.vector(1, 2, 3)).iterator
    it.hasNext should be(true)
    it.next(3) should be(Tensor.vector(1, 2, 3))
    it.hasNext should be(false)
    the [IllegalArgumentException] thrownBy {
      it.next(3)
    } should have message "requirement failed: dataset has no elements left"
  }
}
