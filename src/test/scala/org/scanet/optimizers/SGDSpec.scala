package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.Model

class SGDSpec extends AnyFlatSpec with Matchers {

  "SGD" should "calculate optimization step" in {
    val model = Model[Float, Float, Float]((_, x) => x * x)
    val x = Tensor.scalar(50.0f).const
    val delta = SGD(rate = 0.1).delta(model(50.0f.const, x), x).delta.eval
    delta should be(Tensor.scalar(10f))
  }
}
