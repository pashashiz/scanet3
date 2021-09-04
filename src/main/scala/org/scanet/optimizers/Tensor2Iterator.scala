package org.scanet.optimizers

import org.scanet.core.{Shape, Tensor, TensorType}
import org.scanet.math.Numeric

// low-level mutable implementation
// NOTE: maybe rewrite to have generic N slices...
class Tensor2Iterator[A: TensorType: Numeric](
    private val rows: Iterator[Array[A]],
    private val batch: Int,
    private val splitAt: Int => Int,
    private val withPadding: Boolean)
    extends Iterator[(Tensor[A], Tensor[A])] {

  private val buff: BufferedIterator[Array[A]] = rows.buffered
  val columns: Int = buff.headOption.map(_.length).getOrElse(0)
  val columnsX: Int = splitAt(columns)
  val columnsY: Int = columns - splitAt(columns)

  override def hasNext: Boolean = buff.hasNext

  override def next(): (Tensor[A], Tensor[A]) = {
    if (columns == 0) {
      (Tensor.zeros[A](), Tensor.zeros[A]())
    } else {
      val powerX = batch * columnsX
      val powerY = batch * columnsY
      var x: Array[A] = Array.ofDim[A](powerX)(TensorType[A].classTag)
      var y: Array[A] = Array.ofDim[A](powerY)(TensorType[A].classTag)
      var cursor = 0
      while (buff.hasNext && cursor < batch) {
        val (nextX, nextY) = buff.next().splitAt(columnsX)
        Array.copy(nextX, 0, x, cursor * columnsX, columnsX)
        Array.copy(nextY, 0, y, cursor * columnsY, columnsY)
        cursor = cursor + 1
      }
      if (cursor < batch) {
        if (withPadding) {
          (addPadding(x, cursor * columnsX, powerX), addPadding(y, cursor * columnsY, powerY))
        } else {
          x = x.slice(0, cursor * columnsX)
          y = y.slice(0, cursor * columnsY)
        }
      }
      (
        Tensor[A](x, Shape(x.length / columnsX, columnsX)),
        Tensor[A](y, Shape(y.length / columnsY, columnsY)))
    }
  }

  private def addPadding(x: Array[A], from: Int, until: Int): Unit = {
    val size = until - from
    val zeroX = Array.fill[A](size)(Numeric[A].zero)(TensorType[A].classTag)
    Array.copy(zeroX, 0, x, from, size)
  }
}

object Tensor2Iterator {
  def apply[A: TensorType: Numeric](
      rows: Iterator[Array[A]],
      batch: Int,
      splitAt: Int => Int = _ - 1,
      withPadding: Boolean = true): Tensor2Iterator[A] =
    new Tensor2Iterator(rows, batch, splitAt, withPadding)
}
