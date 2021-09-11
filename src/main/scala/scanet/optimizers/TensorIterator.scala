package scanet.optimizers

import scanet.core.{Shape, Tensor, TensorType}
import scanet.math.Numeric

// low-level mutable implementation
class TensorIterator[A: TensorType: Numeric](
    private val rows: Iterator[Array[A]],
    private val batch: Int,
    private val withPadding: Boolean)
    extends Iterator[Tensor[A]] {

  private val buff: BufferedIterator[Array[A]] = rows.buffered
  val columns: Int = buff.headOption.map(_.length).getOrElse(0)

  override def hasNext: Boolean = buff.hasNext

  override def next(): Tensor[A] = {
    if (columns == 0) {
      Tensor.zeros[A]()
    } else {
      val power = batch * columns
      var array: Array[A] = Array.ofDim[A](power)(TensorType[A].classTag)
      var cursor = 0
      while (buff.hasNext && cursor < batch) {
        Array.copy(buff.next(), 0, array, cursor * columns, columns)
        cursor = cursor + 1
      }
      if (cursor < batch) {
        if (withPadding) {
          val padding = power - cursor * columns
          val zero = Array.fill[A](padding)(Numeric[A].zero)(TensorType[A].classTag)
          Array.copy(zero, 0, array, cursor * columns, padding)
        } else {
          array = array.slice(0, cursor * columns)
        }
      }
      Tensor[A](array, Shape(array.length / columns, columns))
    }
  }
}

object TensorIterator {
  def apply[A: TensorType: Numeric](
      rows: Iterator[Array[A]],
      batch: Int,
      withPadding: Boolean = true): TensorIterator[A] =
    new TensorIterator(rows, batch, withPadding)
}
