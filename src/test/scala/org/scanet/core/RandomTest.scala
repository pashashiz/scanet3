package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Generator.uniform
import org.scanet.syntax.core._

class RandomTest extends AnyFlatSpec with Matchers {

  "uniform distribution" should "work for Int with one next" in {
    val r1 = Random[Int](uniform(1L))
    val (r2, v1) = r1.next
    val (_, v2) = r2.next
    v1 should be(384748)
    v2 should be(-1151252339)
  }

  "uniform distribution" should "work for Int with multiple next" in {
    val x = Random[Int](uniform(1L))
    println(x)
    val (_, values) = Random[Int](uniform(1L)).next(3)
    values should be(Array(384748, -1151252339, -549383847))
  }

}
