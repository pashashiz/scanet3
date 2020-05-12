package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.math.syntax._

class OptimizerSpec extends AnyFlatSpec with Matchers {

  "SDG" should "minimize x^2" in {
    val `x^2` = TensorFunctionBuilder[Float, Float, Float](_ => TensorFunction(x => x * x))
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SDG(rate = 0.1))
      .initWith(Tensor.scalar(5.0f))
      .on(EmptyDataset())
      .epochs(20)
      .doOnEach(step => println(s"result: ${step.result.eval}"))
      .build
    println(opt.run())
  }
}
