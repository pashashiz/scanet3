package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.Regression
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}
import org.scanet.models.Math._

class SGDSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Plain SGD" should "minimize x^2" in {
    val ds = zero
    val opt = Optimizer
      .minimize(`x^2`)
      .using(SGD(rate = 0.1f))
      .initWith(_ => Tensor.scalar(5.0f))
      .on(ds)
      .stopAfter(100.epochs)
      .build
    val x = opt.run()
    val y = `x^2`.loss.compile()(Tensor.scalar(0.0f), x)
    y.toScalar should be <= 0.1f
  }

  "Plain SGD" should "minimize linear regression" in {
    val ds = linearFunction
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD())
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(97)
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.loss.compile()
    val result = regression(BatchingIterator(ds.collect.iterator, 97).next(), weights)
    // note: that reaches 4.5 in 1500 epochs
    result.toScalar should be <= 5.5f
  }

  "SGD with momentum" should "minimize linear regression" in {
    val ds = linearFunction
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.1f))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(97)
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.loss.compile()
    val result = regression(BatchingIterator(ds.collect.iterator, 97).next(), weights)
    // note: that reaches 4.5 in 1500 epochs
    result.toScalar should be <= 5.6f
  }

  "SGD with Nesterov acceleration" should "minimize linear regression" in {
    val ds = linearFunction
    val weights = Optimizer
      .minimize(Regression.linear)
      .using(SGD(momentum = 0.1f, nesterov = true))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(97)
      .stopAfter(100.epochs)
      .build
      .run()
    val regression = Regression.linear.loss.compile()
    val result = regression(BatchingIterator(ds.collect.iterator, 97).next(), weights)
    // note: that reaches 4.5 in 1500 epochs
    result.toScalar should be <= 5.6f
  }
}
