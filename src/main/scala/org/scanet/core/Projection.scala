package org.scanet.core

import org.scanet.core.Slice.syntax.{::, _}

case class Projection(slices: List[Slice], prune: Int = 0) {

  def head: Slice = slices.head
  def tail: Projection = Projection(slices.tail)
  def isEmpty: Boolean = slices.isEmpty
  def rank: Int = slices.size
  def power: Int = shapePruned.power

  def adjustTo(shape: Shape): Projection = {
    require(shape.isInBound(this),
      s"projection $this is out of bound, should fit shape $shape")
    val filledSlices = shape.dims
      .zip(alignRight(shape.rank, ::.build).slices)
      .map { case (shapeSize: Int, slice: Slice) => {
        if (slice.isOpenedRight) (slice.from until shapeSize).build else slice}
      }
    Projection(filledSlices)
  }

  def shapeFull: Shape = Shape(slices.map(_.size))
  def shapePruned: Shape = shapeFull.prune(prune)

  def alignLeft(size: Int, using: Slice): Projection = align(size, using, left = true)
  def alignRight(size: Int, using: Slice): Projection = align(size, using, left = false)
  def align(size: Int, using: Slice, left: Boolean): Projection = {
    if (rank < size) {
      val dimsToFill = size - rank
      val filledSlices = if (dimsToFill > 0) {
        if (left) {
          (0 until dimsToFill).map(_ => using) ++ slices
        } else {
          slices ++ (0 until dimsToFill).map(_ => using)
        }
      } else {
        slices
      }
      Projection(filledSlices.toList)
    } else {
      this
    }
  }

  def canPrune: Int = shapePruned.canPrune
  def prune(max: Int): Projection = copy(prune = prune + math.max(canPrune, max))

  // (*, *, *) :> (1, 2-4, *) = (1, 2-4, *)
  // (1, 2-4, *) :> (1, 2-4) = (1, 1, 2-4)
  def narrow(other: Projection): Projection = {
    require(other.rank == rank,
      s"given projection's rank ${other.rank} does not match to $rank rank")
    val narrowedSlices = slices.zip(other.slices)
      .map { case (sliceThis: Slice, sliceOther: Slice) => sliceThis narrow sliceOther}
    copy(slices = narrowedSlices)
  }
  override def toString: String = {
    val parts = slices.zipWithIndex.map {
      case (slice, index) => (if (index < prune) "`" else "") + slice.toString
    }
    s"(${parts.mkString(", ")})"
  }
}

object Projection {

  def apply[A: CanBuildSliceFrom](a: A): Projection =
    Projection(List(a.build))
  def apply[A: CanBuildSliceFrom, B: CanBuildSliceFrom](a: A, b: B): Projection =
    Projection(List(a.build, b.build))
  def apply[A: CanBuildSliceFrom, B: CanBuildSliceFrom, C: CanBuildSliceFrom](a: A, b: B, c: C): Projection =
    Projection(List(a.build, b.build, c.build))
  def apply[A: CanBuildSliceFrom, B: CanBuildSliceFrom, C: CanBuildSliceFrom, D: CanBuildSliceFrom](a: A, b: B, c: C, d: D): Projection =
    Projection(List(a.build, b.build, c.build, d.build))

  def apply(slices: Seq[Slice]): Projection = new Projection(slices.toList)

  def of(shape: Shape): Projection =
    Projection(shape.dims.map(dim => (0 until dim).build))

  def fill(rank: Int, value: Int): Projection =
    Projection((0 until rank).map(_ => value.build))
}