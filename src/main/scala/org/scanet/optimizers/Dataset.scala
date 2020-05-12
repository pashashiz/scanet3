package org.scanet.optimizers

import org.scanet.core.{Shape, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

trait Dataset[A] {
  def iterator: Iterator[A]
}

trait Iterator[A] {
  def hasNext: Boolean
  def next(batch: Int): Tensor[A]
}

case class TensorDataset[A: TensorType: Numeric](src: Tensor[A]) extends Dataset[A] {
  override def iterator: Iterator[A] = new Iterator[A] {
    var pos: Int = 0
    val size: Int = src.shape.dims.head
    override def hasNext: Boolean = src.shape.dims.head > pos
    override def next(batch: Int): Tensor[A] = {
      require(hasNext, "dataset has no elements left")
      val slice: Tensor[A] = src(pos until math.min(pos + batch, size))
      pos = pos + slice.shape.dims.head
      slice
    }
  }
}

case class NoopDataset[A: TensorType: Numeric]() extends Dataset[A] {
  override def iterator: Iterator[A] = new Iterator[A] {
    var completed = false
    override def hasNext: Boolean = !completed
    override def next(batch: Int): Tensor[A] = {
      completed = true
      Tensor.zeros[A](Shape())
    }
  }
}
