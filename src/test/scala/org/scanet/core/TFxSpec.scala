package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._
import org.scanet.core.Session.syntaxX._

class TFxSpec extends AnyFlatSpec with CustomMatchers {

  "arg" should "work" in {
    println(SeqLike[Seq].asSeq(List(1, 2, 3)))
    println(SeqLike[Id].asSeq(1))
  }

  "TF1" should "work with Id arg" in {
    /*_*/
    val f = TFx1((arg: Id[Output[Int]]) => (10.const + arg).toId).returns[Id[Tensor[Int]]]
    Session.withing(session => {
      val plus10 = f.compile(session)
      println(plus10(Tensor.scalar(5)))
    })
    /*_*/
  }

  "TF1" should "work with Seq arg" in {
    /*_*/
    val plusAll = TFx1((arg: Seq[Output[Int]]) => Seq(plus(arg: _*))).returns[Seq[Tensor[Int]]]
    Session.withing(session => {
      println(plusAll.compile(session).apply(Seq(Tensor.scalar(1), Tensor.scalar(3), Tensor.scalar(5))))
    })
    /*_*/
  }
}
