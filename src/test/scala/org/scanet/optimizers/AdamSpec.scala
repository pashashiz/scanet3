package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.datasets.CSVDataset
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.test.CustomMatchers

class AdamSpec extends AnyFlatSpec with CustomMatchers {

  "Adam" should "minimize linear regression" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(Adam(rate = 0.1))
      .initWith(Tensor.zeros(_))
      .on(ds)
      .each(1.epochs, logResult())
      .stopAfter(300.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }
}