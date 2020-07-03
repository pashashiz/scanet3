package org.scanet.models

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.math.syntax._
import org.scanet.test.CustomMatchers

class RegressionSpec extends AnyFlatSpec with CustomMatchers {

  "linear regression" should "calculate loss with Float precision" in {
    val loss = LinearRegression.withLoss(MeanSquaredError).loss[Float].compile()
    val x = Tensor.matrix(Array(1.0f, 2.0f), Array(2.0f, 4.0f))
    val y = Tensor.matrix(Array(6.0f), Array(12.0f))
    val weights = Tensor.vector(1.0f, 2.0f, 3.0f)
    loss(x, y, weights) should be(Tensor.scalar(17f))
  }

  it should "calculate loss with Double precision" in {
    val loss = LinearRegression.withLoss(MeanSquaredError).loss[Double].compile()
    val x = Tensor.matrix(Array(1.0, 2.0), Array(2.0, 4.0))
    val y = Tensor.matrix(Array(6.0), Array(12.0))
    val weights = Tensor.vector(1.0, 2.0, 3.0)
    loss(x, y, weights) should be(Tensor.scalar(17))
  }

  it should "calculate result" in {
    val result = LinearRegression.result[Float].compile()
    val x = Tensor.matrix(Array(1.0f, 2.0f), Array(2.0f, 4.0f))
    val y = Tensor.matrix(Array(9.0f), Array(17.0f))
    val weights = Tensor.vector(1.0f, 2.0f, 3.0f)
    result(x, weights) should be(y)
  }

  it should "calculate gradient" in {
    val grad = LinearRegression.withLoss(MeanSquaredError).grad[Float].compile()
    val x = Tensor.matrix(Array(1.0f, 2.0f), Array(2.0f, 4.0f))
    val y = Tensor.matrix(Array(6.0f), Array(12.0f))
    val weights = Tensor.vector(0.0f, 0.0f, 0.0f)
    grad(x, y, weights) should be(Tensor.vector(-18.0f, -30.0, -60.0f))
  }

  it should "produce unique toString to be used as a cache key" in {
    LinearRegression.toString should be("LinearRegression")
  }

  "logistic regression" should "calculate loss" in {
    val regression = LogisticRegression.withLoss(BinaryCrossentropy).loss[Float].compile()
    val x = Tensor.matrix(Array(0.34f, 0.78f), Array(0.6f, 0.86f))
    val y = Tensor.matrix(Array(0.402f), Array(0.47800002f))
    val weights = Tensor.vector(0.1f, 0.2f, 0.3f)
    regression(x, y, weights) should be(Tensor.scalar(0.74228245f))
  }

  it should "calculate result" in {
    val result = LinearRegression.result[Float].compile()
    val x = Tensor.matrix(Array(0.34f, 0.78f), Array(0.6f, 0.86f))
    val y = Tensor.matrix(Array(0.402f), Array(0.47800002f))
    val weights = Tensor.vector(0.1f, 0.2f, 0.3f)
    result(x, weights) should be(y)
  }

  it should "calculate gradient " in {
    val grad = LogisticRegression.withLoss(BinaryCrossentropy).grad[Float].compile()
    val x = Tensor.matrix(Array(0.34f, 0.78f), Array(0.6f, 0.86f))
    val y = Tensor.matrix(Array(0.402f), Array(0.47800002f))
    val weights = Tensor.vector(0.1f, 0.2f, 0.3f)
    grad(x, y, weights) should be(Tensor.vector(0.16822177f, 0.075301215f, 0.13678399f))
  }

  it should "produce unique toString to be used as a cache key" in {
    LogisticRegression.toString should be("LogisticRegression")
  }
}

