package scanet.core

import simulacrum.typeclass

import scala.annotation.nowarn

case class Slice(from: Int = 0, until: Int = -1, isRange: Boolean) {
  require(from < until || until == -1, s"from $from should be less than until: $until")
  def isOpenedBoth: Boolean = isOpenedLeft && isOpenedRight
  def isOpenedLeft: Boolean = from == 0
  def isOpenedRight: Boolean = until == -1
  def isSingle: Boolean = !isRange
  def size: Int = until - from
  def elements: Seq[Int] = from until until
  // (4-10) narrow (1-2) -> (5, 6)
  def narrow(other: Slice): Slice = {
    val start = from + other.from
    val end = if (isOpenedRight) {
      other.until
    } else if (other.isOpenedBoth) {
      until
    } else {
      math.min(from + other.until, until)
    }
    Slice(start, end, isRange && other.isRange)
  }
  override def toString: String = {
    if (isSingle) {
      from.toString
    } else {
      val start = if (isOpenedLeft) ":" else from.toString
      val end = if (isOpenedRight) ":" else until.toString
      if (start == ":" || end == ":") {
        s"$start$end"
      } else {
        s"$start-$end"
      }
    }
  }
}

@nowarn @typeclass trait CanBuildSliceFrom[A] {
  def build(a: A): Slice
}

trait SingleSlice extends CanBuildSliceFrom[Int] {
  override def build(a: Int): Slice = {
    Slice(a, a + 1, isRange = false)
  }
}

trait RangeSlice extends CanBuildSliceFrom[Range] {
  override def build(a: Range): Slice = Slice(a.start, a.end, isRange = true)
}

case class Unbound()

trait UnboundSlice extends CanBuildSliceFrom[Unbound] {
  override def build(a: Unbound): Slice = Slice(0, -1, isRange = true)
}

object Slice {
  trait Instances {
    implicit def singleSlice: CanBuildSliceFrom[Int] = new SingleSlice {}
    implicit def rangeSlice: CanBuildSliceFrom[Range] = new RangeSlice {}
    implicit def unboundSlice: CanBuildSliceFrom[Unbound] = new UnboundSlice {}
  }
  trait AllSyntax extends Instances with CanBuildSliceFrom.ToCanBuildSliceFromOps
  object syntax extends AllSyntax {
    def :: : Unbound = Unbound()
  }
}
