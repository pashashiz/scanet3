package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.Tensor
import org.scanet.datasets.EmptyDataset
import org.scanet.math.syntax._
import org.scanet.models.Math._

class OptimizerSpec extends AnyFlatSpec with Matchers {

  "SGD" should "minimize x^2" in {
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SGD(rate = 0.1))
      .initWith(Tensor.scalar(5.0f))
      .on(EmptyDataset())
      .epochs(20)
      //.doEach(step => println(s"result: ${step.result()}"))
      .build
    val x = opt.run()
    println(`x^2`.result.compile()(Tensor.scalar(0.0f), x))
  }

  it should "minimize x^2 with Nesterov acceleration" in {
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SGD(rate = 0.1, momentum = 0.9, nesterov = true))
      .initWith(Tensor.scalar(5.0f))
      .on(EmptyDataset())
      .epochs(20)
      //.doEach(step => println(s"result: ${step.result()}"))
      .build
    val x = opt.run()
    println(`x^2`.result.compile()(Tensor.scalar(0.0f), x))
  }
}
