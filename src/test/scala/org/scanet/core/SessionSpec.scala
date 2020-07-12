package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.withing
import org.scanet.core.Tensor.scalar
import org.scanet.syntax.placeholder
import org.scanet.test.CustomMatchers
import org.scanet.math.syntax._

class SessionSpec extends AnyFlatSpec with CustomMatchers {

  "session" should "eval a single output" in {
    withing(session => {
      session.runner.evalU(10.const) should be(scalar(10))
    })
  }

  it should "eval a sequence of outputs" in {
    withing(session => {
      // NOTE: we need to explicitly pass OutputSeq[Int]
      // which is an alias for Seq[Output[Int]]
      session.runner.evalU[OutputSeq[Int]](Seq(5.const, 10.const)) should
        be(Seq(scalar(5), scalar(10)))
    })
  }

  it should "eval a tuple2 of outputs" in {
    withing(session => {
      session.runner.evalU((5.const, 10.const)) should
        be((scalar(5), scalar(10)))
    })
  }

  it should "eval a tuple2 with one output and sequence of outputs" in {
    withing(session => {
      session.runner.evalU[(Output[Int], OutputSeq[Int])]((1.const, Seq(5.const, 10.const))) should
        be((scalar(1), Seq(scalar(5), scalar(10))))
    })
  }

  it should "eval a tuple3 of outputs" in {
    withing(session => {
      session.runner.evalU((1.const, 5f.const, 10.const)) should
        be((scalar(1), scalar(5f), scalar(10)))
    })
  }

  "session" should "eval generically when single output is used" in {
    withing(session => {
      session.runner.evalX[Id[Output[Int]], Id[Tensor[Int]]](10.const) should be(scalar(10))
    })
  }

  it should "eval generically when tuple 2 is used" in {
    withing(session => {
      session.runner.evalX[(Id[Output[Int]], Id[Output[Int]]), (Id[Tensor[Int]], Id[Tensor[Int]])](
        (10.const, 5.const)) should be((scalar(10), scalar(5)))
    })
  }

  it should "eval generically when tuple 3 is used" in {
    withing(session => {
      session.runner.evalX[(Id[Output[Int]], Id[Output[Int]], Id[Output[Int]]), (Id[Tensor[Int]], Id[Tensor[Int]], Id[Tensor[Int]])](
        (10.const, 5.const, 1.const)) should be((scalar(10), scalar(5), scalar(1)))
    })
  }

  "placeholder" should "be substituted with a session" in {
    withing(session => {
      val a = placeholder[Int]()
      val b = 10.const
      val c = a + b
      session.runner
        .feed(a -> scalar(5))
        .eval(c) should be(scalar(15))
    })
  }

  "session pool" should "work" in {
    val pool = SessionPool.max(4)
    pool.withing(session => {
      session.runner.eval(5.const) should be(scalar(5))
    })
  }

  it should "reuse existing sessions" in {
    val pool = SessionPool.max(4)
    val prevSession = pool.withing(identity)
    pool.withing(session => session should be theSameInstanceAs prevSession)
  }

  it should "create a new session on demand" in {
    val pool = SessionPool.max(4)
    pool.withing(session1 => {
      pool.used should be(1)
      pool.withing(session2 => {
        pool.used should be(2)
        session1 should not be theSameInstanceAs(session2)
      })
    })
  }
}
