package org.scanet.datasets

import org.scanet.core.{Projection, Shape, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

import scala.io.Source

trait Dataset[X] {
  def iterator: Iterator[X]
}

trait Iterator[X] {
  def hasNext: Boolean
  def next(batch: Int): Tensor[X]
  def size: Int
  def batches(batch: Int): Int = {
    val partialBatch = if (size % batch > 0) 1 else 0
    (size / batch) + partialBatch
  }
  def shape: Shape
}

private[datasets] trait DataSource[X] {
  def size: Int
  def slice(from: Int, to: Int): Tensor[X]
  def shape: Shape
}

private[datasets] abstract class IndexedDataset[X](val ds: DataSource[X]) extends Dataset[X] {
  override def iterator: Iterator[X] = new Iterator[X] {
    private var index: Int = 0

    override def hasNext: Boolean = size > index

    override def next(batchSize: Int): Tensor[X] = {
      require(hasNext, "dataset has no elements left")
      val pos = index
      index = index + batchSize
      if (pos + batchSize > size) {
        ds.slice(math.max(size - batchSize, 0), size)
      } else {
        ds.slice(pos, pos + batchSize)
      }
    }

    override def size: Int = ds.size

    def shape: Shape = ds.shape
  }
}

case class TensorDataset[X: TensorType : Numeric](src: Tensor[X]) extends IndexedDataset(new DataSource[X] {
  val size: Int = src.shape.dims.head

  override def slice(from: Int, to: Int): Tensor[X] = src(from until to)

  val shape: Shape = src.shape.remove(0)
})

case class EmptyDataset[X: TensorType : Numeric]() extends IndexedDataset(new DataSource[X] {
  val size: Int = 1

  override def slice(from: Int, to: Int): Tensor[X] = Tensor.zeros[X](Shape())

  val shape: Shape = Shape()
})

case class CSVDataset(path: String) extends IndexedDataset(new DataSource[Float] {
  private lazy val data: Vector[Array[Float]] = {
    Source.fromInputStream(getClass.getClassLoader.getResourceAsStream(path))
      .getLines()
      .map(_.split(",").map(_.toFloat))
      .toVector
  }
  private lazy val columns: Int = data(0).length

  lazy val size: Int = data.size

  override def slice(from: Int, to: Int): Tensor[Float] = {
    val slice = data.slice(from, to)
    Tensor(slice.flatten.toArray, Shape(slice.size, columns))
  }

  lazy val shape: Shape = Shape(columns)
})
