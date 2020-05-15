package org.scanet.datasets

import org.scanet.core.{Shape, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

import scala.io.Source

trait Dataset[X] {
  def iterator: Iterator[X]
}

trait Iterator[X] {
  def hasNext: Boolean
  def next(batch: Int): Tensor[X]
  def size: Option[Int]
}

case class TensorDataset[X: TensorType: Numeric](src: Tensor[X]) extends Dataset[X] {
  override def iterator: Iterator[X] = new Iterator[X] {
    var pos: Int = 0
    val size: Option[Int] = Some(src.shape.dims.head)
    override def hasNext: Boolean = src.shape.dims.head > pos
    override def next(batch: Int): Tensor[X] = {
      require(hasNext, "dataset has no elements left")
      val slice: Tensor[X] = src(pos until math.min(pos + batch, size.get))
      pos = pos + slice.shape.dims.head
      slice
    }
  }
}

case class EmptyDataset[X: TensorType: Numeric]() extends Dataset[X] {
  override def iterator: Iterator[X] = new Iterator[X] {
    var completed = false
    override def size: Option[Int] = Some(0)
    override def hasNext: Boolean = !completed
    override def next(batch: Int): Tensor[X] = {
      completed = true
      Tensor.zeros[X](Shape())
    }
  }
}

case class CSVDataset(path: String) extends Dataset[Float] {

  override def iterator: Iterator[Float] = new Iterator[Float] {
    private var pos: Int = 0
    private val data: Vector[Array[Float]] = {
      Source.fromInputStream(getClass.getClassLoader.getResourceAsStream(path))
        .getLines()
        .map(_.split(",").map(_.toFloat))
        .toVector
    }
    var columns: Int = data(0).length
    var size: Option[Int] = Some(data.size)

    override def hasNext: Boolean = size.get > pos
    override def next(batch: Int): Tensor[Float] = {
      require(hasNext, "dataset has no elements left")
      val slice = data.slice(pos, math.min(pos + batch, size.get))
      pos = pos + slice.size
      val x = slice.flatten.toArray
      Tensor(x, Shape(slice.size, columns))
    }
  }
}
