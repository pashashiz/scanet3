package scanet.core

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Session.withing
import scanet.core.Tensor.scalar
import scanet.math.syntax._
import scanet.syntax.placeholder
import scanet.test.CustomMatchers

import scala.collection.immutable.Seq

class SessionSpec extends AnyFlatSpec with CustomMatchers {

  "session" should "eval a single output" in {
    withing { session =>
      session.runner.eval(10.const) should be(scalar(10))
    }
  }

  it should "eval a sequence of outputs" in {
    withing { session =>
      session.runner.eval[Seq[Expr[Int]]](Seq(5.const, 10.const)) should
      be(Seq(scalar(5), scalar(10)))
    }
  }

  it should "eval a tuple2 of outputs" in {
    withing { session =>
      session.runner.eval((5.const, 10.const)) should
      be((scalar(5), scalar(10)))
    }
  }

  it should "eval a tuple2 with one output and sequence of outputs" in {
    withing { session =>
      println(session.devices)
      session.runner.eval[(Expr[Int], Seq[Expr[Int]])]((1.const, Seq(5.const, 10.const))) should
      be((scalar(1), Seq(scalar(5), scalar(10))))
    }
  }

  it should "eval a tuple3 of outputs" in {
    withing { session =>
      session.runner.eval((1.const, 5f.const, 10.const)) should
      be((scalar(1), scalar(5f), scalar(10)))
    }
  }

  "placeholder" should "be substituted with a session" in {
    withing { session =>
      val a = placeholder[Int]()
      val b = 10.const
      val c = a + b
      session.runner.feed(a -> scalar(5)).eval(c) should be(scalar(15))
    }
  }

  "session pool" should "work" in {
    val pool = SessionPool.max(4)
    pool.within { session =>
      session.runner.eval(5.const) should be(scalar(5))
    }
  }

  it should "reuse existing sessions" in {
    val pool = SessionPool.max(4)
    val prevSession = pool.within(identity)
    pool.within { session => session should be theSameInstanceAs prevSession }
  }

  it should "create a new session on demand" in {
    val pool = SessionPool.max(4)
    pool.within { session1 =>
      pool.used should be(1)
      pool.within { session2 =>
        pool.used should be(2)
        session1 should not be theSameInstanceAs(session2)
      }
    }
  }
}
