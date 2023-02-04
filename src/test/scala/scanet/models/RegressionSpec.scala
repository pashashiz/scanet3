package scanet.models

import org.scalatest.flatspec.AnyFlatSpec
import scanet.core.Tensor
import scanet.math.syntax._
import scanet.models.Loss._
import scanet.test.CustomMatchers
import scala.collection.immutable.Seq

class RegressionSpec extends AnyFlatSpec with CustomMatchers {

  "linear regression" should "calculate loss with Float precision" in {
    val loss = LinearRegression().withLoss(MeanSquaredError).loss[Float].compile
    val x = Tensor.matrix(Array(1.0f, 2.0f), Array(2.0f, 4.0f))
    val y = Tensor.matrix(Array(6.0f), Array(12.0f))
    val weights = Tensor.matrix(Array(2.0f), Array(3.0f))
    val bias = Tensor.vector(1.0f)
    loss(x, y, Seq(weights, bias)) should be(Tensor.scalar(17f))
  }

  it should "calculate loss with Double precision" in {
    val loss = LinearRegression().withLoss(MeanSquaredError).loss[Double].compile
    val x = Tensor.matrix(Array(1.0, 2.0), Array(2.0, 4.0))
    val y = Tensor.matrix(Array(6.0), Array(12.0))
    val weights = Tensor.matrix(Array(2.0), Array(3.0))
    val bias = Tensor.vector(1.0)
    loss(x, y, Seq(weights, bias)) should be(Tensor.scalar(17))
  }

  it should "calculate result" in {
    val result = LinearRegression().result[Float].compile
    val x = Tensor.matrix(Array(1.0f, 2.0f), Array(2.0f, 4.0f))
    val y = Tensor.matrix(Array(9.0f), Array(17.0f))
    val weights = Tensor.matrix(Array(2.0f), Array(3.0f))
    val bias = Tensor.vector(1.0f)
    result(x, Seq(weights, bias)) should be(y)
  }

  it should "calculate gradient" in {
    val grad = LinearRegression().withLoss(MeanSquaredError).grad[Float].compile
    val x = Tensor.matrix(Array(1.0f, 2.0f), Array(2.0f, 4.0f))
    val y = Tensor.matrix(Array(6.0f), Array(12.0f))
    val weights = Tensor.matrix(Array(0.0f), Array(0.0f))
    val bias = Tensor.vector(0.0f)
    grad(x, y, Seq(weights, bias)) should be(Seq(
      Tensor.matrix(Array(-30.0f), Array(-60.0f)),
      Tensor.vector(-18.0f)))
  }

  it should "produce unique toString to be used as a cache key" in {
    LinearRegression().toString should be("Dense(1,Zero,Zeros) >> Bias(1,Zero,Zeros)")
  }

  "logistic regression" should "calculate loss" in {
    val regression = LogisticRegression().withLoss(BinaryCrossentropy).loss[Float].compile
    val x = Tensor.matrix(Array(0.34f, 0.78f), Array(0.6f, 0.86f))
    val y = Tensor.matrix(Array(0.402f), Array(0.47800002f))
    val weights = Tensor.matrix(Array(0.2f), Array(0.3f))
    val bias = Tensor.vector(0.1f)
    regression(x, y, Seq(weights, bias)).toScalar should be(0.7422824f +- 1e-6f)
  }

  it should "calculate result" in {
    val result = LogisticRegression().result[Float].compile
    val x = Tensor.matrix(Array(0.34f, 0.78f), Array(0.6f, 0.86f))
    val y = Tensor.matrix(Array(0.599168f), Array(0.617276f))
    val weights = Tensor.matrix(Array(0.2f), Array(0.3f))
    val bias = Tensor.vector(0.1f)
    val predicted = result(x, Seq(weights, bias))
    predicted.const.roundAt(6).eval should be(y)
  }

  it should "calculate gradient " in {
    val grad = LogisticRegression().withLoss(BinaryCrossentropy).grad[Float].compile
    val x = Tensor.matrix(Array(0.34f, 0.78f), Array(0.6f, 0.86f))
    val y = Tensor.matrix(Array(0.402f), Array(0.47800002f))
    val weights = Tensor.matrix(Array(0.2f), Array(0.3f))
    val bias = Tensor.vector(0.1f)
    val result = grad(x, y, Seq(weights, bias))
      .map(_.const.roundAt(6).eval)
    result should be(Seq(
      Tensor.matrix(Array(0.075301f), Array(0.136784f)),
      Tensor.vector(0.168222f)))
  }

  it should "produce unique toString to be used as a cache key" in {
    LogisticRegression().toString should be("Dense(1,Zero,Zeros) >> Bias(1,Zero,Zeros) >> Sigmoid")
  }
}
