package org.scanet.optimizers

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.models.LinearRegression
import org.scanet.optimizers.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}
import org.scanet.models.Math.`x^2`
import org.scanet.optimizers.Effect.logResult

class SGDSpec extends AnyFlatSpec with CustomMatchers with SharedSpark with Datasets {

  "Plain SGD" should "minimize x^2" in {
    val ds = zero
    val trained = Optimizer
      .minimize(`x^2`[Float])
      .using(SGD(rate = 0.1f))
      .initWith(_ => Tensor.scalar(5.0f))
      .on(ds)
      .stopAfter(100.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    // x = weights
    val y = loss(Tensor.scalar(0.0f), Tensor.scalar(0.0f))
    y.toScalar should be <= 0.1f
  }

  "Plain SGD" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = Optimizer
      .minimize(LinearRegression[Float])
      .using(SGD())
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(97)
      .stopAfter(100.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    // note: that reaches 4.5 in 1500 epochs
    loss(x, y).toScalar should be <= 5.5f
  }

  "SGD with momentum" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = Optimizer
      .minimize(LinearRegression[Float])
      .using(SGD(momentum = 0.1f))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(97)
      .stopAfter(100.epochs)
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    // note: that reaches 4.5 in 1500 epochs
    loss(x, y).toScalar should be <= 5.6f
  }

  "SGD with Nesterov acceleration" should "minimize linear regression" in {
    val ds = linearFunction
    val trained = Optimizer
      .minimize(LinearRegression[Float])
      .using(SGD(momentum = 0.1f, nesterov = true))
      .initWith(s => Tensor.zeros(s))
      .on(ds)
      .batch(97)
      .stopAfter(100.epochs)
      .each(1.epochs, logResult())
      .build
      .run()
    val loss = trained.loss.compile()
    val (x, y) = Tensor2Iterator(ds.collect.iterator, 97).next()
    // note: that reaches 4.5 in 1500 epochs
    loss(x, y).toScalar should be <= 5.6f
  }
}
