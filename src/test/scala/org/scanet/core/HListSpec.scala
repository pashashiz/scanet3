package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import shapeless._
import shapeless.ops.hlist.Mapper
import shapeless.poly._

class HListSpec extends AnyFlatSpec {

  "hlist" should "work" in {
    val sets: Set[Int] :: Set[String] :: HNil = Set(1) :: Set("foo") :: HNil

    val list = sets.toList
    println(list)
    println(toOptions(sets))
  }

  object choose extends (Set ~> Option) {
    def apply[T](s : Set[T]): Option[T] = s.headOption
  }

  def toOptions[L <: HList, Mapped <: HList](hlist: L)(
    implicit m: Mapper.Aux[choose.type, L, Mapped]): Mapped = {
    hlist.map(choose)
  }

  // we need to have (Output(A1), Output(A2), ...) -> (Tensor(A1), Tensor(A2), ...)
  // Product[Output[..]] ~> Product[Tensor[..]]

}
