package scanet.core

import scanet.core.syntax.intTfTypeInst
import scanet.core.syntax.longTfTypeInst

import scala.annotation.nowarn

case class Shape(dims: List[Int]) extends Ordered[Shape] {

  require(dims.forall(_ > 0), "dimension size cannot be 0")
  val power: Int = dims.product
  val dimsPower: List[Int] = {
    val powers = dims.foldLeft(List(power))((power, dim) => power.head / dim :: power)
    powers.reverse.tail
  }

  def indexOf(absPosition: Int): List[Int] = {
    val (indexes, _) = dimsPower.foldLeft((List[Int](), absPosition))((acc, dimPower) => {
      val (indexes, prevPos) = acc
      val index = prevPos / dimPower
      val nextPos = prevPos % dimPower
      (index :: indexes, nextPos)
    })
    indexes.reverse
  }

  def get(dim: Int): Int = if (dim == -1) last else dims(dim)

  def apply(dim: Int): Int = get(dim)

  def rank: Int = dims.size

  def axes: List[Int] = dims.indices.toList

  def axesExcept(other: Int*): List[Int] = {
    val indexedAxes = indexAxes(other)
    (dims.indices.toSet -- indexedAxes.toSet).toList.sorted
  }

  def isScalar: Boolean = rank == 0

  def isInBound(projection: Projection): Boolean = {
    val rankInRange = projection.rank <= rank
    val numOutOfBounds = projection.slices.zip(dims)
      .map {
        case (slice: Slice, max: Int) =>
          if (slice.isOpenedRight) 0 else slice.until - max
      }
      .count(_ > 0)
    rankInRange && numOutOfBounds == 0
  }

  def head: Int = dims.head

  def tail: Shape = Shape(dims.tail)

  def take(n: Int): Shape = Shape(dims.take(n))
  def takeRight(n: Int): Shape = Shape(dims.takeRight(n))

  def drop(n: Int): Shape = Shape(dims.drop(n))
  def dropRight(n: Int): Shape = Shape(dims.dropRight(n))

  def last: Int = {
    require(!isScalar, "cannot get last dimension for scalar")
    dims.last
  }

  def prepend(dim: Int): Shape = Shape(dim +: dims: _*)
  def +:(dim: Int): Shape = prepend(dim)

  def append(dim: Int): Shape = Shape(dims :+ dim: _*)
  def :+(dim: Int): Shape = append(dim)

  def ++(right: Shape): Shape = Shape(dims ++ right.dims)

  def splitAt(index: Int): (Shape, Shape) = {
    require(index < rank, s"index should be < $rank")
    val (left, right) = dims.splitAt(index)
    (Shape(left), Shape(right))
  }

  def splitAtHalf: (Shape, Shape) = {
    require(rank % 2 == 0, s"rank should be even but was $rank")
    splitAt(rank / 2)
  }

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
      val filledDims =
        if (dimsToFill > 0) {
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

  @nowarn def >>>(size: Int, using: Int): Shape = alignLeft(rank + size, using)
  def >>>(size: Int): Shape = >>>(size, 1)
  @nowarn def <<<(size: Int, using: Int): Shape = alignRight(rank + size, using)
  def <<<(size: Int): Shape = <<<(size, 1)

  def >>(size: Int): Shape = {
    require(rank >= size, s"cannot $this >> $size, cause rank should be >= $size")
    Shape(dims.dropRight(size))
  }
  def <<(size: Int): Shape = {
    require(rank >= size, s"cannot $this << $size, cause rank should be >= $size")
    Shape(dims.drop(size))
  }

  def canPrune: Int = dims.takeWhile(_ == 1).size
  def pruneAll: Shape = Shape(dims.dropWhile(_ == 1))
  def prune(max: Int): Shape = {
    val (_, pruned) = dims.foldLeft((max, List[Int]()))((acc, dim) => {
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
      dims.zip(alignedOther.dims).forall { case (g, s) => s == 1 || g == s }
    }
  }

  def broadcastableAny(other: Shape): Boolean =
    broadcastableBy(other) || other.broadcastableBy(this)

  def broadcastableAxes(other: Shape): Seq[Int] = {
    require(broadcastableAny(other), s"cannot find broadcastable axes for $this and $other")
    if (rank < other.rank) {
      Seq()
    } else {
      val alignedOther = other.alignLeft(rank, 1)
      alignedOther.dims.zipWithIndex
        .filter { case (dim, _) => dim == 1 }
        .map { case (_, index) => index }
    }
  }

  def maxDims(other: Shape): Shape = {
    val maxRank = rank max other.rank
    val left = alignLeft(maxRank, 1)
    val right = other.alignLeft(maxRank, 1)
    val dimsResult = left.dims.zip(right.dims)
      .map { case (l, r) => l max r }
    Shape(dimsResult)
  }

  def permute(axes: Int*): Shape = {
    val indexedAxes = indexAxes(axes)
    require(
      rank == indexedAxes.size,
      "the number of permutation indexes " +
      s"should be equal to rank $rank, but was (${axes.mkString(", ")})")
    Shape(indexedAxes.foldLeft(List[Int]())((permDims, index) => dims(index) :: permDims).reverse)
  }

  def select(axes: Int*): Shape = {
    val indexedAxes = indexAxes(axes)
    require(
      indexedAxes.forall(i => i < rank && i >= 0),
      s"the number of selected axes " +
      s"should be less or equal to rank $rank, but was (${axes.mkString(", ")})")
    Shape(indexedAxes.map(get).toList)
  }

  def remove(axes: Int*): Shape = {
    val indexedAxes = indexAxes(axes)
    require(
      indexedAxes.forall(i => i < rank && i >= 0),
      s"the number of removed axes " +
      s"should be less or equal to rank $rank, but was (${axes.mkString(", ")})")
    val filteredDims = dims.zipWithIndex
      .filter {
        case (_, i) =>
          !indexedAxes.contains(i)
      }
      .map { case (dim, _) => dim }
    Shape(filteredDims)
  }

  def updated(axis: Int, value: Int): Shape = updateAll(value)(axis)

  def updateAll(value: Int)(axes: Int*): Shape = {
    val indexedAxes = indexAxes(axes)
    require(
      indexedAxes.forall(i => i < rank && i >= 0),
      s"the number of updated axes " +
      s"should be less or equal to rank $rank, but was (${axes.mkString(", ")})")
    val updatedDims = dims.zipWithIndex.map {
      case (dim, i) =>
        if (indexedAxes.contains(i)) value else dim
    }
    Shape(updatedDims)
  }

  def updateAllExcept(value: Int)(axes: Int*): Shape = {
    val indexedAxes = indexAxes(axes)
    val axesToUpdate = dims.indices.toSet -- indexedAxes.toSet
    updateAll(value)(axesToUpdate.toList: _*)
  }

  def indexAxes(axes: Seq[Int]): Seq[Int] =
    axes.map(indexAxis)

  def indexAxis(axis: Int): Int =
    if (axis == -1) dims.size - 1 else axis

  def minus(other: Shape): Shape = {
    require(broadcastableAny(other), s"cannot $this - $other")
    if (endsWith(other)) {
      Shape(dims.take(rank - other.rank))
    } else {
      Shape()
    }
  }

  def -(other: Shape): Shape = minus(other)

  def toArray: Array[Int] = dims.toArray
  def toLongArray: Array[Long] = dims.map(_.toLong).toArray

  def toTensor: Tensor[Int] = Tensor.vector(toArray)
  def toLongTensor: Tensor[Long] = Tensor.vector(toLongArray)

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
