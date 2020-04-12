package org.scanet.linalg

import org.scanet.linalg.Slice.syntax._

case class View(src: IndexesSource, originalShape: Shape, projection: Projection) {

  private val shapeFull: Shape = projection.shapeFull
  val shape: Shape = projection.shapeShort

  def isScalar: Boolean = shape.isScalar

  def narrow(other: Projection): View = {
    require(other.rank <= shape.rank,
      s"projection $other has rank '${other.rank}' which is greater than " +
        s"shape's rank '${shape.rank}'")
    val adjusted = other
      .alignRight(shape.rank, ::.build)
      .adjustTo(shape)
      .alignLeft(originalShape.rank, 0.build)
    require(shapeFull.isInBound(adjusted),
      s"projection $other is out of bound, should fit shape $shape")
    View(src, originalShape, projection narrow adjusted)
  }

  def reshape(into: Shape): View = {
    require(shape.power == into.power,
      s"shape $shape cannot be reshaped into $into")
    if (originalShape.power == shape.power && shape.power == projection.power) {
      View(into)
    } else {
      View(ViewSource(this), into)
    }
  }

  def positions: Array[Int] = {
    val indexes = src.indexes
    def loop(dims: Seq[(Slice, Int)], absPos: Int, acc: Array[Int], seqPos: Int): (Array[Int], Int) = {
      if (dims.isEmpty) {
        acc(seqPos) = indexes(absPos)
        (acc, seqPos + 1)
      } else {
        val (slice, dimPower) = dims.head
        slice.elements.foldLeft((acc, seqPos))((a, i) => {
          val nextIndex = absPos + dimPower * i
          loop(dims.tail, nextIndex, a._1, a._2)
        })
      }
    }
    val dims = projection.slices.zip(originalShape.dimsPower)
    val (array, _) = loop(dims, 0, Array.ofDim(shape.power), 0)
    array
  }

  override def toString: String = {
    val text = s"$originalShape x $projection = $shape"
    src match {
      case ViewSource(view) => s"$view -> $text"
      case IdentitySource() => text
    }
  }
}

trait IndexesSource {
  def indexes: Int => Int
}

sealed case class ViewSource(view: View) extends IndexesSource {
  override def indexes: Int => Int = view.positions
}

sealed case class IdentitySource() extends IndexesSource {
  override def indexes: Int => Int = identity[Int]
}

object View {

  def apply(shape: Shape): View = View(shape, Projection.of(shape))
  def apply(shape: Shape, projection: Projection): View =
    View(IdentitySource(), shape, projection)
  def apply(indexes: IndexesSource, shape: Shape): View =
    new View(indexes, shape, Projection.of(shape))
  def apply(indexes: IndexesSource, shape: Shape, projection: Projection): View =
    new View(indexes, shape, projection.adjustTo(shape))
}