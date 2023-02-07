package scanet.math.grad

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Session.withing
import scanet.core.Tensor.scalar
import scanet.math.syntax._

import scala.jdk.CollectionConverters._
import scala.collection.immutable.Seq

class GradSpec extends AnyWordSpec with Matchers {

  "gradient" when {
    "taken in respect to multiple inputs" should {
      "work" in {
        val a = 3f.const.as("a")
        val b = 4f.const.as("b")
        val f = (a * b).pow(3)
        val grads = f.grad(Seq(a, b)).returns[Float]
        grads.eval should be(Seq(scalar(1728f), scalar(1296f)))
      }
    }
  }
}
