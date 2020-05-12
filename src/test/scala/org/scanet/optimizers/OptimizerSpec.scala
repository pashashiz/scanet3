package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scanet.core.{Output, Tensor}
import org.scanet.math.syntax._

class OptimizerSpec extends AnyFlatSpec with Matchers {

  "SDG" should "minimize x^2" in {
    val x2 = new TensorFunctionBuilder[Float, Float, Float] {
      override def apply(v1: Output[Float]): TensorFunction[Float, Float] = TensorFunction[Float, Float](x => x * x)
    }
    val optimizer = Optimizer[Float, Float, Float](
      alg = SDG(rate = 0.1),
      funcBuilder = x2,
      initArgs = Tensor.scalar(5.0f),
      dataset = NoopDataset(),
      epochs = 20)
    println(optimizer.minimize())
  }
}
