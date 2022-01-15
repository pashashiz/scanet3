package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Tensor
import scanet.models.Activation._
import scanet.models.Loss.MeanSquaredError
import scanet.models.Regularization.L2
import scanet.syntax._
import scanet.test.{CustomMatchers, Datasets, SharedSpark}
import scala.collection.immutable.Seq

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
        Array(1f, 0.1f, 1f),
        Array(0.5f, 1f, 0f),
        Array(1f, 1f, 0.2f),
        Array(0.1f, 1f, 0.3f))
      val b1 = Tensor.vector(0f, 0f, 0f, 0f)
      val w2 = Tensor.matrix(
        Array(0.1f, 0.5f, 1f, 0f))
      val b2 = Tensor.vector(0f)
      val result = model.result[Float].compile()
      val expected = Tensor.matrix(
        Array(0.7053571f),
        Array(0.7701361f),
        Array(0.7627533f),
        Array(0.8018865f))
      result(x, Seq(w1, b1, w2, b2)) should be(expected)
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
    model.penalty(Seq(w1.const, b1.const, w2.const, b2.const)).eval should be(Tensor.scalar(3.83f))
  }

  "calculate right loss" in {
    val model = Dense(3, Sigmoid) >> Dense(1, Sigmoid)
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(0f))
    val w1 = Tensor.matrix(
      Array(1f, 0.1f, 1f),
      Array(0.5f, 1f, 0f),
      Array(1f, 1f, 0.2f))
    val b1 = Tensor.vector(0f, 0f, 0f)
    val w2 = Tensor.matrix(
      Array(0.1f, 0.5f, 1f))
    val b2 = Tensor.vector(0f)
    val loss = model.withLoss(MeanSquaredError).loss[Float].compile()
    loss(x, y, Seq(w1, b1, w2, b2)) should be(Tensor.scalar(0.33996207f))
  }

  "calculate right loss with penalty" in {
    val model = Dense(3, Sigmoid, L2(1f)) >> Dense(1, Sigmoid, L2(1f))
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f))
    val y = Tensor.matrix(Array(1f), Array(0f))
    val w1 = Tensor.matrix(
      Array(1f, 0.1f, 1f),
      Array(0.5f, 1f, 0f),
      Array(1f, 1f, 0.2f))
    val b1 = Tensor.vector(0f, 0f, 0f)
    val w2 = Tensor.matrix(
      Array(0.1f, 0.5f, 1f))
    val b2 = Tensor.vector(0f)
    val loss = model.withLoss(MeanSquaredError).loss[Float].compile()
    loss(x, y, Seq(w1, b1, w2, b2)) should be(Tensor.scalar(3.6199622f))
  }
}
