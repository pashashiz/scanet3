package scanet.optimizers

import scanet.core.{Monoid, Shape, Tensor, TensorType}
import scanet.optimizers.Iterators._

object Iterators {

  sealed trait Remaining
  case object Partial extends Remaining
  case object PadZeros extends Remaining
  case object Skip extends Remaining

  trait AllSyntax {
    implicit def toResourceIteratorOps[A](it: Iterator[A]): ResourceIteratorOps[A] =
      new ResourceIteratorOps[A](it)
  }

  object syntax extends AllSyntax
}

class ResourceIterator[A](it: Iterator[A], cleanUp: () => Unit)
    extends Iterator[A]
    with AutoCloseable {
  override def hasNext: Boolean =
    if (it.hasNext) {
      true
    } else {
      close(); false
    }
  override def next(): A = it.next()
  override def close(): Unit = cleanUp()
}

class ResourceIteratorOps[A](val it: Iterator[A]) extends AnyVal {
  def onClose(f: () => Unit): ResourceIterator[A] = new ResourceIterator[A](it, f)
  def using(resource: AutoCloseable): ResourceIterator[A] = onClose(() => resource.close())
}

class TensorIterator[A: Monoid](
    private val rows: Iterator[Record[A]],
    private val shapes: (Shape, Shape),
    private val batch: Int,
    private val remaining: Remaining)
    extends Iterator[(Tensor[A], Tensor[A])] {

  private val buff: rows.GroupedIterator[Record[A]] =
    rows.sliding(batch, batch).withPartial(remaining == PadZeros || remaining == Partial)

  val (shapeX: Shape, shapeY: Shape) = shapes
  private val sizeX = shapeX.power
  private val sizeY = shapeY.power
  private val sizeBatchX = batch * sizeX
  private val sizeBatchY = batch * sizeY

  override def hasNext: Boolean = buff.hasNext

  override def next(): (Tensor[A], Tensor[A]) = {
    val x = Array.ofDim[A](sizeBatchX)(TensorType[A].classTag)
    val y = Array.ofDim[A](sizeBatchY)(TensorType[A].classTag)
    if (buff.hasNext) {
      val records = buff.next().iterator
      var cursor = 0
      while (records.hasNext && cursor < batch) {
        val Record(nextX, nextY) = records.next()
        Array.copy(nextX, 0, x, cursor * sizeX, sizeX)
        Array.copy(nextY, 0, y, cursor * sizeY, sizeY)
        cursor = cursor + 1
      }
      remaining match {
        case Partial if cursor < batch =>
          (
            Tensor[A](x.slice(0, cursor * sizeX), cursor +: shapeX),
            Tensor[A](y.slice(0, cursor * sizeY), cursor +: shapeY))
        case _ =>
          (
            Tensor[A](x, batch +: shapeX),
            Tensor[A](y, batch +: shapeY))
      }
    } else {
      throw new NoSuchElementException()
    }
  }
}

object TensorIterator {
  def apply[A: Monoid](
      rows: Iterator[Record[A]],
      shapes: (Shape, Shape),
      batch: Int,
      remaining: Remaining = Skip): TensorIterator[A] =
    new TensorIterator(rows, shapes, batch, remaining)
}
