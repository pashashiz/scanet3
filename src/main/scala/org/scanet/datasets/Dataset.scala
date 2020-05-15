package org.scanet.datasets

import org.scanet.core.{Projection, Shape, Tensor, TensorType}
import org.scanet.math.Numeric
import org.scanet.math.syntax._

import scala.io.Source
import scala.reflect.ClassTag

trait Dataset[X] {
  def iterator: Iterator[X]
  def shape(batch: Int): Shape
  def size: Int
}

trait Iterator[X] {
  def hasNext: Boolean
  def next(batch: Int): Tensor[X]
}

case class TensorDataset[X: TensorType: Numeric: ClassTag](src: Tensor[X]) extends Dataset[X] {
  override def iterator: Iterator[X] = new Iterator[X] {
    var pos: Int = 0
    override def hasNext: Boolean = src.shape.dims.head > pos
    override def next(batch: Int): Tensor[X] = {
      require(hasNext, "dataset has no elements left")
      val slice =
        if (batch > size) src(pos until size)
        else if (pos + batch <= size) src(pos until pos + batch)
        else {
          val zero = Numeric[X].zero
          val reminder = (pos + batch - size) * src.shape.tail.power
          val array = src(pos until size).toArray appendedAll Array.fill(reminder)(zero)
          Tensor(array, shape(batch))
        }
      pos = pos + slice.shape.dims.head
      slice
    }
  }
  val size: Int = src.shape.dims.head
  override def shape(batch: Int): Shape = {
    val maxSize = if (batch < size) batch else size
    src.view.narrow(Projection(0 until maxSize)).shape
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

  override def shape(batch: Int): Shape = Shape()
  val size: Int = 0
}

case class CSVDataset(path: String) extends Dataset[Float] {

  private lazy val data: Vector[Array[Float]] = {
    Source.fromInputStream(getClass.getClassLoader.getResourceAsStream(path))
      .getLines()
      .map(_.split(",").map(_.toFloat))
      .toVector
  }
  private lazy val columns: Int = data(0).length

  lazy val size: Int = data.size

  override def iterator: Iterator[Float] = new Iterator[Float] {
    private var pos: Int = 0

    override def hasNext: Boolean = size > pos
    override def next(batch: Int): Tensor[Float] = {
      require(hasNext, "dataset has no elements left")
      val slice =
        if (batch > size) data
        else if (pos + batch <= size) data.slice(pos, pos + batch)
        else {
          val reminder = Vector.fill(pos + batch - size)(Array.fill(columns)(0f))
          data.slice(pos, size) appendedAll reminder
        }
      pos = pos + batch
      val x = slice.flatten.toArray
      Tensor(x, Shape(slice.size, columns))
    }
  }

  override def shape(batch: Int): Shape = Shape(if (batch < size) batch else size, columns)
}
