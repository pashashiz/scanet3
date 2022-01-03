package scanet.optimizers

import scanet.core.{Monoid, Shape, Tensor, TensorType}

// low-level mutable implementation
// NOTE: maybe rewrite to have generic N slices...
class TensorIterator[A: Monoid](
    private val rows: Iterator[Record[A]],
    private val shapes: (Shape, Shape),
    private val batch: Int,
    private val withPadding: Boolean)
    extends Iterator[(Tensor[A], Tensor[A])] {

  private val buff: BufferedIterator[Record[A]] = rows.buffered
  val (shapeX: Shape, shapeY: Shape) = shapes
  private val sizeX = shapeX.power
  private val sizeY = shapeY.power

  override def hasNext: Boolean = buff.hasNext

  override def next(): (Tensor[A], Tensor[A]) = {
    val powerX = batch * sizeX
    val powerY = batch * sizeY
    var x: Array[A] = Array.ofDim[A](powerX)(TensorType[A].classTag)
    var y: Array[A] = Array.ofDim[A](powerY)(TensorType[A].classTag)
    var cursor = 0
    while (buff.hasNext && cursor < batch) {
      val Record(nextX, nextY) = buff.next()
      Array.copy(nextX, 0, x, cursor * sizeX, sizeX)
      Array.copy(nextY, 0, y, cursor * sizeY, sizeY)
      cursor = cursor + 1
    }
    if (cursor < batch) {
      if (withPadding) {
        (addPadding(x, cursor * sizeX, powerX), addPadding(y, cursor * sizeY, powerY))
      } else {
        x = x.slice(0, cursor * sizeX)
        y = y.slice(0, cursor * sizeY)
      }
    }
    (
      Tensor[A](x, Shape(x.length / sizeX, sizeX)),
      Tensor[A](y, Shape(y.length / sizeY, sizeY)))
  }

  private def addPadding(x: Array[A], from: Int, until: Int): Unit = {
    val size = until - from
    val zeroX = Array.fill[A](size)(Monoid[A].zero)(TensorType[A].classTag)
    Array.copy(zeroX, 0, x, from, size)
  }
}

object TensorIterator {
  def apply[A: Monoid](
      rows: Iterator[Record[A]],
      shapes: (Shape, Shape),
      batch: Int,
      withPadding: Boolean = true): TensorIterator[A] =
    new TensorIterator(rows, shapes, batch, withPadding)
}
