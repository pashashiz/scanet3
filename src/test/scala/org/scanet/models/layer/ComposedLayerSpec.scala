package org.scanet.models.layer

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.models.Activation._
import org.scanet.syntax._
import org.scanet.test.{CustomMatchers, Datasets, SharedSpark}

class ComposedLayerSpec extends AnyFlatSpec with CustomMatchers  with SharedSpark with Datasets {

  "layers composition" should "produce right result (forward pass)" in {
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
      Array(0.705357f),
      Array(0.7701362f),
      Array(0.76275325f),
      Array(0.80188656f))
    result(x, Seq(w1, w2)) should be(expected)
  }
}
