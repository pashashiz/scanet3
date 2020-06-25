package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.Effect.logResult
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class AMSGradSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "AMSGrad" should "minimize linear regression" in {
    val ds = linearFunction
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(AMSGrad(rate = 0.1f))
      .initWith(Tensor.zeros(_))
      .on(ds)
      .batch(97)
      .each(1.epochs, logResult())
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.loss.compile()
    val result = regression(BatchingIterator(ds.collect.iterator, 97).next(), weights)
    result.toScalar should be <= 4.5f
  }
}
