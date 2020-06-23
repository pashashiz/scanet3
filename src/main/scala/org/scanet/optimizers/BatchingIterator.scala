package org.scanet.optimizers

import org.scanet.core.{Shape, Tensor, TensorType}
import org.scanet.math.Numeric

// todo: infer columns
class BatchingIterator[A: TensorType: Numeric](val rows: Iterator[Array[A]], val batch: Int, val columns: Int) extends Iterator[Tensor[A]] {

  override def hasNext: Boolean = rows.hasNext

  override def next(): Tensor[A] = {
    // low-level mutable implementation
    val power = batch * columns
    val array = Array.ofDim[A](power)(TensorType[A].classTag)
    var cursor = 0
    while (rows.hasNext && cursor < power) {
      Array.copy(rows.next(), 0, array, cursor, columns)
      cursor = cursor + columns
    }
    if (cursor < power) {
      val padding = power - cursor
      val zero = Array.fill[A](padding)(Numeric[A].zero)(TensorType[A].classTag)
      Array.copy(zero, 0, array, cursor, padding)
    }
    Tensor[A](array, Shape(batch, columns))
  }
}

object BatchingIterator {
  def apply[A: TensorType: Numeric](rows: Iterator[Array[A]], batch: Int, columns: Int): BatchingIterator[A] =
    new BatchingIterator(rows, batch, columns)
}