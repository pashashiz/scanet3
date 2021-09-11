package scanet.math.grad

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.OutputSeq
import scanet.core.Session.withing
import scanet.core.Tensor.scalar
import scanet.math.syntax._

import scala.collection.JavaConverters._
import scala.collection.immutable.Seq

class GradSpec extends AnyWordSpec with Matchers {

  "gradient" when {
    "taken in respect to multiple inputs" should {
      "work" in {
        val a = 3f.const.as("a")
        val b = 4f.const.as("b")
        val f = (a * b).pow(3)
        val grads: OutputSeq[Float] = f.grad(Seq(a, b)).returns[Float]
        grads.eval should be(Seq(scalar(1728f), scalar(1296f)))
      }
      "reuse common operations of a computation graph" in {
        val a = 3f.const.as("a")
        val b = 4f.const.as("b")
        val f = (a * b).pow(3)
        val ga = f.grad(a).returns[Float].as("grad_a")
        val gb = f.grad(b).returns[Float].as("grad_b")
        // in our case both grads will reuse
        // 3 * (a * b)^2 sub-graph
        (ga, gb).eval should be((scalar(1728f), scalar(1296f)))
        withing(session => {
          val graph = session.toGraph(Seq(ga, gb))
          val ops = graph.operations().asScala
          ops should have size 11
        })
      }
    }
  }
}
