package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._

class TensorFunctionSpec extends AnyFlatSpec with Matchers {

  "tensor function" should "compute result and gradient" in {

    // batch = 100
    // sum(f)


    val x2 = TensorFunction[Float, Float](x => x * x)
    x2(5f.const).eval should be(Tensor.scalar(25.0f))
    x2.grad(5f.const).eval should be(Tensor.scalar(10.0f))
  }
}
