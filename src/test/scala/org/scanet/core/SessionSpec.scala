package org.scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Session.withing
import org.scanet.core.Tensor.scalar
import org.scanet.math.syntax._
import org.scanet.syntax.placeholder
import org.scanet.test.CustomMatchers

import scala.collection.JavaConverters._

class SessionSpec extends AnyFlatSpec with CustomMatchers {

  "session" should "eval a single output" in {
    withing(session => {
      session.runner.eval(10.const) should be(scalar(10))
    })
  }

  it should "eval a sequence of outputs" in {
    withing(session => {
      // NOTE: we need to explicitly pass OutputSeq[Int]
      // which is an alias for Seq[Output[Int]]
      session.runner.eval[OutputSeq[Int]](Seq(5.const, 10.const)) should
        be(Seq(scalar(5), scalar(10)))
    })
  }

  it should "eval a tuple2 of outputs" in {
    withing(session => {
      session.runner.eval((5.const, 10.const)) should
        be((scalar(5), scalar(10)))
    })
  }

  it should "eval a tuple2 with one output and sequence of outputs" in {
    withing(session => {
      session.runner.eval[(Output[Int], OutputSeq[Int])]((1.const, Seq(5.const, 10.const))) should
        be((scalar(1), Seq(scalar(5), scalar(10))))
    })
  }

  it should "eval a tuple3 of outputs" in {
    withing(session => {
      session.runner.eval((1.const, 5f.const, 10.const)) should
        be((scalar(1), scalar(5f), scalar(10)))
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

  "session" should "reuse the same already existing sub-graphs" in {
    val a = 5f.const
    // here a.sqrt should be done just once
    // like _a = a.sqr; x = _a + _a
    val x = a.sqr + a.sqr
    x.eval should be(scalar(50f))
    withing(session => {
      val graph = session.toGraph(Seq(x))
      val ops = graph.operations().asScala
      // const(5), pow, const(2), add | NOTE: only 1 pow op
      ops should have size 4
    })
  }
}
