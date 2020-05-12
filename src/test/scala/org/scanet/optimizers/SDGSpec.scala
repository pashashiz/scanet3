package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._

class SDGSpec extends AnyFlatSpec with Matchers {

  "SDG" should "calculate optimization step" in {
    val x2 = TensorFunction[Float, Float](x => x * x)
    println(SDG(rate = 0.1).delta(x2, 50.0f.const).eval)
  }
}
