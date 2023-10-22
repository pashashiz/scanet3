package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Params.Weights
import scanet.core.Path._
import scanet.core.{Params, Shape, Tensor}
import scanet.models.Activation._
import scanet.models.Loss.MeanSquaredError
import scanet.models.Regularization.L2
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}

class ComposedLayerSpec extends AnyWordSpec with CustomMatchers with SharedSpark with Datasets {

  "layers composition" should {

    "calculate right forward pass" in {
      val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
      val x = Tensor.matrix(
        Array(0f, 0f, 1f),
        Array(0f, 1f, 1f),
        Array(1f, 0f, 1f),
        Array(1f, 1f, 1f))
      val w1 = Tensor.matrix(
        Array(1f, 0.5f, 1f, 0.1f),
        Array(0.1f, 1f, 1f, 1f),
        Array(1f, 0f, 0.2f, 0.3f))
      val b1 = Tensor.vector(0f, 0f, 0f, 0f)
      val w2 = Tensor.matrix(
        Array(0.1f),
        Array(0.5f),
        Array(1f),
        Array(0f))
      val b2 = Tensor.vector(0f)
      val forward = model.result[Float].compile
      val expected = Tensor.matrix(
        Array(0.705357f),
        Array(0.770136f),
        Array(0.762753f),
        Array(0.801886f))
      val params = Params(
        0 / Weights -> w1,
        1 / Weights -> b1,
        3 / Weights -> w2,
        4 / Weights -> b2)
      val result = forward(x, params).const.roundAt(6).eval
      result should be(expected)
    }
  }

  "calculate right penalty pass with regularization" in {
    val model = Dense(4, Sigmoid, reg = L2(1f)) >> Dense(1, Sigmoid, reg = L2(1f))
    val w1 = Tensor.matrix(
      Array(1f, 0.1f, 1f),
      Array(0.5f, 1f, 0f),
      Array(1f, 1f, 0.2f),
      Array(0.1f, 1f, 0.3f))
    val b1 = Tensor.vector(0f, 0f, 0f, 0f)
    val w2 = Tensor.matrix(
      Array(0.1f, 0.5f, 1f, 0f))
    val b2 = Tensor.vector(0f)
    val params = Params(
      0 / Weights -> w1,
      1 / Weights -> b1,
      3 / Weights -> w2,
      4 / Weights -> b2)
    model.penalty(Shape(1, 4), params.mapValues(_.const)).eval should be(
      Tensor.scalar(3.83f))
  }

  "calculate right loss" in {
    val model = Dense(3, Sigmoid) >> Dense(1, Sigmoid)
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(0f))
    val w1 = Tensor.matrix(
      Array(1f, 0.5f, 1f),
      Array(0.1f, 1f, 1f),
      Array(1f, 0f, 0.2f))
    val b1 = Tensor.vector(0f, 0f, 0f)
    val w2 = Tensor.matrix(
      Array(0.1f),
      Array(0.5f),
      Array(1f))
    val b2 = Tensor.vector(0f)
    val params = Params(
      0 / Weights -> w1,
      1 / Weights -> b1,
      3 / Weights -> w2,
      4 / Weights -> b2)
    val loss = model.withLoss(MeanSquaredError).loss[Float].compile
    val result = loss(x, y, params).const.roundAt(6).eval
    result should be(Tensor.scalar(0.339962f))
  }

  "calculate right loss with penalty" in {
    val model = Dense(3, Sigmoid, L2(1f)) >> Dense(1, Sigmoid, L2(1f))
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(0f))
    val w1 = Tensor.matrix(
      Array(1f, 0.5f, 1f),
      Array(0.1f, 1f, 1f),
      Array(1f, 0f, 0.2f))
    val b1 = Tensor.vector(0f, 0f, 0f)
    val w2 = Tensor.matrix(
      Array(0.1f),
      Array(0.5f),
      Array(1f))
    val b2 = Tensor.vector(0f)
    val params = Params(
      0 / Weights -> w1,
      1 / Weights -> b1,
      3 / Weights -> w2,
      4 / Weights -> b2)
    val loss = model.withLoss(MeanSquaredError).loss[Float].compile
    loss(x, y, params) should be(Tensor.scalar(3.6199622f))
  }
}
