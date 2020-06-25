package org.scanet.optimizers

import org.scanet.core.{Shape, Tensor, TensorType}
import org.scanet.math.Numeric

// low-level mutable implementation
class BatchingIterator[A: TensorType: Numeric](private val rows: Iterator[Array[A]], private val batch: Int) extends Iterator[Tensor[A]] {

  private val buff: BufferedIterator[Array[A]] = rows.buffered
  val columns: Int = buff.headOption.map(_.length).getOrElse(0)

  override def hasNext: Boolean = buff.hasNext

  override def next(): Tensor[A] = {
    val power = batch * columns
    val array: Array[A] = Array.ofDim[A](power)(TensorType[A].classTag)
    var cursor = 0
    while (buff.hasNext && cursor < batch) {
      Array.copy(buff.next(), 0, array, cursor * columns, columns)
      cursor = cursor + 1
    }
    if (cursor < batch) {
      val padding = power - cursor * columns
      val zero = Array.fill[A](padding)(Numeric[A].zero)(TensorType[A].classTag)
      Array.copy(zero, 0, array, cursor * columns, padding)
    }
    if (columns == 0) Tensor.zeros[A]() else Tensor[A](array, Shape(batch, columns))
  }
}

object BatchingIterator {
  def apply[A: TensorType: Numeric](rows: Iterator[Array[A]], batch: Int): BatchingIterator[A] =
    new BatchingIterator(rows, batch)
}