package scanet.math.stat

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.math.syntax._

import scala.collection.immutable.Seq

class KernelsSpec extends AnyWordSpec with Matchers {

  "variance" should {

    val tensor = Tensor.matrix(Array(1f, 1.3f, 1.1f), Array(4f, 5f, 4.6f)).const

    "calculate variance across all axis by default" in {
      tensor.variance.roundAt(2).eval should be(Tensor.scalar(2.98f))
    }

    "support reducing along matrix columns" in {
      tensor.variance(Seq(0)).roundAt(2).eval should be(Tensor.vector(2.25f, 3.42f, 3.06f))
    }

    "support reducing along matrix rows" in {
      tensor.variance(Seq(1)).roundAt(2).eval should be(Tensor.vector(0.02f, 0.17f))
    }

    "reduce and keep dimensions" in {
      tensor.variance(Seq(1), keepDims = true).roundAt(2).eval should be(
        Tensor.matrix(Array(0.02f), Array(0.17f)))
    }
  }

  "std" should {

    val tensor = Tensor.matrix(Array(1f, 1.3f, 1.1f), Array(4f, 5f, 4.6f)).const

    "calculate standard deviation across all axis by default" in {
      tensor.std.roundAt(2).eval should be(Tensor.scalar(1.73f))
    }

    "support reducing along matrix columns" in {
      tensor.std(Seq(0)).roundAt(2).eval should be(Tensor.vector(1.5f, 1.85f, 1.75f))
    }

    "support reducing along matrix rows" in {
      tensor.std(Seq(1)).roundAt(2).eval should be(Tensor.vector(0.12f, 0.41f))
    }

    "reduce and keep dimensions" in {
      tensor.std(Seq(1), keepDims = true).roundAt(2).eval should be(
        Tensor.matrix(Array(0.12f), Array(0.41f)))
    }
  }
}
