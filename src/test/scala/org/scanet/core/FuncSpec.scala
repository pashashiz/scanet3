package org.scanet.core

import java.util.concurrent.atomic.AtomicInteger

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FuncSpec extends AnyFlatSpec with Matchers {

  "memoization" should "cache result of the function and reuse it" in {
    val counter = new AtomicInteger(0)
    val plus = memoize((a: Int, b: Int) => {
      counter.incrementAndGet()
      a + b
    })
    counter.get() should be(0)
    plus(2, 3) should be(5)
    counter.get() should be(1)
    plus(2, 3) should be(5)
    counter.get() should be(1)
  }
}
