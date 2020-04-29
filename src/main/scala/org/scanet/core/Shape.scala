package org.scanet.core

case class Shape(dims: List[Int]) extends Ordered[Shape] {

  require(dims.forall(_ > 0), "dimension size cannot be 0")
  val power: Int = dims.product
  val dimsPower: List[Int] = {
    val powers = dims.foldLeft(List(power))((power, dim) => power.head / dim :: power)
    powers.reverse.tail
  }

  def indexOf(absPosition: Int): List[Int] = {
    val (indexes, _) = dimsPower.foldLeft((List[Int](), absPosition))(
      (acc, dimPower) => {
        val (indexes, prevPos) = acc
        val index = prevPos / dimPower
        val nextPos = prevPos % dimPower
        (index :: indexes, nextPos)
      })
    indexes.reverse
  }

  def get(dim: Int): Int = dims(dim)

  def rank: Int = dims.size

  def isScalar: Boolean = rank == 0

  def isInBound(projection: Projection): Boolean = {
    val rankInRange = projection.rank <= rank
    val numOutOfBounds = projection.slices.zip(dims)
      .map { case (slice: Slice, max: Int) =>
        if (slice.isOpenedRight) 0 else slice.until - max
      }
      .count(_ > 0)
    rankInRange && numOutOfBounds == 0
  }

  def head: Int = dims.head

  def tail: Shape = Shape(dims.tail)

  def last: Int = dims.last

  def alignLeft(size: Int, using: Int): Shape = align(size, using, left = true)
  def alignRight(size: Int, using: Int): Shape = align(size, using, left = false)
  def align(size: Int, using: Int, left: Boolean): Shape = {
    if (rank < size) {
      val dimsToFill = size - rank
      val filledDims = if (dimsToFill > 0) {
        if (left) {
          (0 until dimsToFill).map(_ => using) ++ dims
        } else {
          dims ++ (0 until dimsToFill).map(_ => using)
        }
      } else {
        dims
      }
      Shape(filledDims.toList)
    } else {
      this
    }
  }

  def canPrune: Int = dims.takeWhile(_ == 1).size
  def pruneAll: Shape = Shape(dims.dropWhile(_ == 1))
  def prune(max: Int): Shape = {
    val (_, pruned) = dims.foldLeft((max, List[Int]()))(
      (acc, dim) => {
        val (toPrune, dims) = acc
        if (toPrune > 0 && dim == 1) {
          (toPrune - 1, dims)
        } else {
          (0, dim :: dims)
        }
      })
    Shape(pruned.reverse)
  }

  def squeeze: Shape = Shape(dims.filter(_ > 1))

  def endsWith(other: Shape): Boolean = dims.endsWith(other.dims)

  def broadcastableBy(smaller: Shape): Boolean = {
    val alignedSmaller = smaller.alignLeft(rank, 1)
    dims.zip(alignedSmaller.dims).forall {case (g, s) => s == 1 || g == s}
  }

  def broadcastableAny(other: Shape): Boolean =
    broadcastableBy(other) || other.broadcastableBy(this)

  def toLongArray: Array[Long] = dims.map(_.toLong).toArray

  override def toString: String = s"(${dims.mkString(", ")})"

  override def compare(that: Shape): Int = {
    if (this.rank == that.rank) 0 else if (this.rank > that.rank) 1 else -1
  }
}

object Shape {

  def apply(dims: Int*): Shape = Shape(dims.toList)

  def of(array: Array[Long]): Shape = Shape(array.map(_.toInt).toList)
}







