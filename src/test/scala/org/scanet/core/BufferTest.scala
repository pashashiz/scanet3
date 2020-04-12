package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.syntax.core._

class BufferTest extends AnyFlatSpec with Matchers {

  "buffer" should "have toString" in {
    Buffer.wrap(Array(1, 2, 3)).position(1).toString should
      be("Buffer[Int](capacity=3, position=1, limit=3, direct=false)[2, 3]")
  }

  it should "be converted into array" in {
    Buffer.wrap(Array(1, 2, 3)).toArray should be(Array(1, 2, 3))
  }

  it should "return next element" in {
    Buffer.wrap(Array(1, 2, 3)).get should be(1)
  }
}
