package org.scanet.datasets

import org.scanet.core.{Shape, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

trait Dataset[X] {
  def iterator: Iterator[X]
}

trait Iterator[X] {
  def hasNext: Boolean
  def next(batch: Int): Tensor[X]
}

case class TensorDataset[X: TensorType: Numeric](src: Tensor[X]) extends Dataset[X] {
  override def iterator: Iterator[X] = new Iterator[X] {
    var pos: Int = 0
    val size: Int = src.shape.dims.head
    override def hasNext: Boolean = src.shape.dims.head > pos
    override def next(batch: Int): Tensor[X] = {
      require(hasNext, "dataset has no elements left")
      val slice: Tensor[X] = src(pos until math.min(pos + batch, size))
      pos = pos + slice.shape.dims.head
      slice
    }
  }
}

case class EmptyDataset[X: TensorType: Numeric]() extends Dataset[X] {
  override def iterator: Iterator[X] = new Iterator[X] {
    var completed = false
    override def hasNext: Boolean = !completed
    override def next(batch: Int): Tensor[X] = {
      completed = true
      Tensor.zeros[X](Shape())
    }
  }
}
