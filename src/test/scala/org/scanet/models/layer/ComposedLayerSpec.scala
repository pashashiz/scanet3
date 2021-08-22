package org.scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import org.scanet.core.Tensor
import org.scanet.models.Activation._
import org.scanet.models.Loss.MeanSquaredError
import org.scanet.models.Regularization.L2
import org.scanet.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class ComposedLayerSpec extends AnyWordSpec with CustomMatchers  with SharedSpark with Datasets {

  "layers composition" should {

    "calculate right forward pass" in {
      val model = Dense(4, Sigmoid) >> Dense(1, Sigmoid)
      val x = Tensor.matrix(
        Array(0f, 0f, 1f),
        Array(0f, 1f, 1f),
        Array(1f, 0f, 1f),
        Array(1f, 1f, 1f))
      val w1 = Tensor.matrix(
        Array(0f, 1f, 0.1f, 1f),
        Array(0f, 0.5f, 1f, 0f),
        Array(0f, 1f, 1f, 0.2f),
        Array(0f, 0.1f, 1f, 0.3f))
      val w2 = Tensor.matrix(
        Array(0f, 0.1f, 0.5f, 1f, 0f))
      val result = model.result[Float].compile()
      val expected = Tensor.matrix(
        Array(0.7053571f),
        Array(0.7701361f),
        Array(0.7627533f),
        Array(0.8018865f))
      result(x, Seq(w1, w2)) should be(expected)
    }
  }

  "calculate right penalty pass with regularization" in {
    val model = Dense(4, Sigmoid, reg = L2(1f)) >> Dense(1, Sigmoid, reg = L2(1f))
    val w1 = Tensor.matrix(
      Array(0f, 1f, 0.1f, 1f),
      Array(0f, 0.5f, 1f, 0f),
      Array(0f, 1f, 1f, 0.2f),
      Array(0f, 0.1f, 1f, 0.3f))
    val w2 = Tensor.matrix(
      Array(0f, 0.1f, 0.5f, 1f, 0f))
    model.penalty(Seq(w1.const, w2.const)).eval should be(Tensor.scalar(3.83f))
  }

  "calculate right loss" in {
    val model = Dense(3, Sigmoid) >> Dense(1, Sigmoid)
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(0f))
    val w1 = Tensor.matrix(
      Array(0f, 1f, 0.1f, 1f),
      Array(0f, 0.5f, 1f, 0f),
      Array(0f, 1f, 1f, 0.2f))
    val w2 = Tensor.matrix(
      Array(0f, 0.1f, 0.5f, 1f))
    val loss = model.withLoss(MeanSquaredError).loss[Float].compile()
    loss(x, y, Seq(w1, w2)) should be(Tensor.scalar(0.33996207f))
  }

  "calculate right loss with penalty" in {
    val model = Dense(3, Sigmoid, L2(1f)) >> Dense(1, Sigmoid, L2(1f))
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(0f))
    val w1 = Tensor.matrix(
      Array(0f, 1f, 0.1f, 1f),
      Array(0f, 0.5f, 1f, 0f),
      Array(0f, 1f, 1f, 0.2f))
    val w2 = Tensor.matrix(
      Array(0f, 0.1f, 0.5f, 1f))
    val loss = model.withLoss(MeanSquaredError).loss[Float].compile()
    loss(x, y, Seq(w1, w2)) should be(Tensor.scalar(3.6199622f))
  }
}
