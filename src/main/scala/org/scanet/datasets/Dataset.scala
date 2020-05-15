package org.scanet.datasets

import org.scanet.core.{Projection, Shape, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

import scala.io.Source

trait Dataset[X] {
  def iterator: Iterator[X]
  def shape(batch: Int): Shape
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

  override def shape(batch: Int): Shape = src.view.narrow(Projection(0 until batch)).shape
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

  override def shape(batch: Int): Shape = Shape()
}

case class CSVDataset(path: String) extends Dataset[Float] {

  private lazy val data: Vector[Array[Float]] = {
    Source.fromInputStream(getClass.getClassLoader.getResourceAsStream(path))
      .getLines()
      .map(_.split(",").map(_.toFloat))
      .toVector
  }

  private lazy val columns: Int = data(0).length

  override def iterator: Iterator[Float] = new Iterator[Float] {
    private var pos: Int = 0

    val size: Option[Int] = Some(data.size)

    override def hasNext: Boolean = size.get > pos
    override def next(batch: Int): Tensor[Float] = {
      require(hasNext, "dataset has no elements left")
      val slice = data.slice(pos, math.min(pos + batch, size.get))
      pos = pos + slice.size
      val x = slice.flatten.toArray
      Tensor(x, Shape(slice.size, columns))
    }
  }

  override def shape(batch: Int): Shape = Shape(batch, columns)
}
