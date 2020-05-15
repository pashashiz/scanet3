package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.datasets.EmptyDataset
import org.scanet.math.syntax._
import org.scanet.models.Model

class OptimizerSpec extends AnyFlatSpec with Matchers {

  "SGD" should "minimize x^2" in {
    val `x^2` = Model[Float, Float, Float]((_, x) => x * x)
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SGD(rate = 0.1))
      .initWith(Tensor.scalar(5.0f))
      .on(EmptyDataset())
      .epochs(20)
      .doOnEach(step => println(s"result: ${step.result.eval}"))
      .build
    val x = opt.run()
    println(`x^2`(0.0f.const, x.const).eval)
  }

  it should "minimize x^2 with Nesterov acceleration" in {
    val `x^2` = Model[Float, Float, Float]((_, x) => x * x)
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SGD(rate = 0.1, momentum = 0.9, nesterov = true))
      .initWith(Tensor.scalar(5.0f))
      .on(EmptyDataset())
      .epochs(20)
      .doOnEach(step => println(s"result: ${step.result.eval}"))
      .build
    val x = opt.run()
    println(`x^2`(0.0f.const, x.const).eval)
  }
}
