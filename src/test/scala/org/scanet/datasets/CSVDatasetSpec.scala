package org.scanet.datasets

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.core.syntax._

class CSVDatasetSpec extends AnyFlatSpec with Matchers {

  "CSV dataset" should "read records" in {
    val ds = CSVDataset("linear_function_1.scv")
    val it = ds.iterator
    it.size should be(Some(97))
    it.next(2) should be(Tensor.matrix(
      Array(6.1101f, 17.592f),
      Array(5.5277f, 9.1302f)))
  }
}
