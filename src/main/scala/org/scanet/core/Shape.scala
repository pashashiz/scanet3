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

  def prepend(dim: Int) = Shape(dim +: dims: _*)
  def +:(dim: Int) = prepend(dim)

  def append(dim: Int) = Shape(dims :+ dim: _*)
  def :+(dim: Int) = append(dim)

  def insert(dim: Int, size: Int): Shape = {
    require(dim >= 0 && dim <= rank, s"couldn't insert dimension $dim cause rank is $rank")
    if (dim < rank) {
      Shape((dims.take(dim) :+ size) ++ dims.slice(dim, dims.size + 1))
    } else {
      Shape(dims :+ size)
    }
  }

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

  def broadcastableBy(other: Shape): Boolean = {
    if (rank < other.rank) {
      false
    } else {
      val alignedOther = other.alignLeft(rank, 1)
      dims.zip(alignedOther.dims).forall {case (g, s) => s == 1 || g == s}
    }
  }

  def broadcastableAny(other: Shape): Boolean =
    broadcastableBy(other) || other.broadcastableBy(this)

  def broadcastableAxises(other: Shape): Seq[Int] = {
    require(broadcastableAny(other), s"cannot find broadcastable axises for $this and $other")
    if (rank < other.rank) {
      Seq()
    } else {
      val alignedOther = other.alignLeft(rank, 1)
      alignedOther.dims.zipWithIndex
        .filter { case (dim, _) => dim == 1 }
        .map { case (_, index) => index }
    }
  }

  def permute(indexes: Int*): Shape = {
    require(rank == indexes.size, "the number of permutation indexes " +
      s"should be equal to rank $rank, but was (${indexes.mkString(", ")})")
    Shape(indexes.foldLeft(List[Int]())((permDims, index) => dims(index) :: permDims).reverse)
  }

  def select(axises: Int*): Shape = {
    require(axises.forall(_ < rank), s"the number of selected axises " +
      s"should be less or equal to rank $rank, but was (${axises.mkString(", ")})")
    Shape(axises.map(dims(_)).toList)
  }

  def remove(axises: Int*): Shape = {
    require(axises.forall(_ < rank), s"the number of removed axises " +
      s"should be less or equal to rank $rank, but was (${axises.mkString(", ")})")
    val filteredDims = dims.zipWithIndex
      .filter {case (_, i) => !axises.contains(i)}
      .map { case (dim, _) => dim }
    Shape(filteredDims)
  }

  def minus(other: Shape): Shape = {
    require(broadcastableAny(other), s"cannot $this - $other")
    if (endsWith(other)) {
      Shape(dims.take(rank - other.rank))
    } else {
      Shape()
    }
  }

  def -(other: Shape): Shape = minus(other)

  def toLongArray: Array[Long] = dims.map(_.toLong).toArray

  override def toString: String = s"(${dims.mkString(", ")})"

  override def compare(that: Shape): Int = {
    if (this.rank == that.rank) 0 else if (this.rank > that.rank) 1 else -1
  }
}

object Shape {

  def apply(dims: Int*): Shape = Shape(dims.toList)

  def of(array: Array[Long]): Shape = Shape(array.map(_.toInt).toList)

  def scalar: Shape = Shape()
}







