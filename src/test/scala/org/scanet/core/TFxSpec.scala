package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._
import org.scanet.core.TFx.syntax._

class TFxSpec extends AnyFlatSpec with CustomMatchers {

  "arg" should "work" in {
    println(SeqLike[Seq].asSeq(List(1, 2, 3)))
    println(SeqLike[Id].asSeq(1))
  }

  "TF1" should "work with Id arg" in {
    val f = new TFx1[Id, Int, Id[Output[Int]], Id[Tensor[Int]]]((shape: Shape) => {
      val p1 = placeholder[Int](shape)
      val out = p1 + 10.const
      (Seq(p1), out)
    })
    Session.withing(session => {
      val plus10 = f.compile(session)
      println(plus10(Tensor.scalar(5)))
    })


    val f2 = TFx1[Id, Int, Id[Output[Int]]]((arg: Id[Output[Int]]) => {
      10.const plus arg
    }).returns[Id[Tensor[Int]]]
  }

  "TF1" should "work with Seq arg" in {
    val plusAll = new TFx1[Seq, Int, Id[Output[Int]], Id[Tensor[Int]]]((shapes: Seq[Shape]) => {
      val placeholders = shapes.map(shape => placeholder[Int](shape))
      (placeholders, plus(placeholders: _*))
    })
    Session.withing(session => {
      println(plusAll.compile(session).apply(Seq(Tensor.scalar(1), Tensor.scalar(3), Tensor.scalar(5))))
    })


    val plusAll2 = TFx1[Seq, Int, Id[Output[Int]]]((arg: Seq[Output[Int]]) => {
      plus(arg: _*)
    }).returns[Id[Tensor[Int]]]
  }
}
