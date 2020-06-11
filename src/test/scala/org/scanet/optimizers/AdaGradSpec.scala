package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.datasets.CSVDataset
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.syntax._
import org.scanet.test.CustomMatchers

class AdaGradSpec extends AnyFlatSpec with CustomMatchers {

  "AdaGrad" should "minimize linear regression" in {
    val ds = CSVDataset("linear_function_1.scv")
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AdaGrad())
      .initWith(Tensor.zeros(_))
      .on(ds)
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.result.compile()
    val result = regression(ds.iterator.next(100), weights)
    result.toScalar should be <= 4.5f
  }
}