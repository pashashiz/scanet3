package org.scanet.models.nn

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.models.Sigmoid
import org.scanet.syntax._
import org.scanet.test.CustomMatchers

class DenseLayer extends AnyFlatSpec with CustomMatchers {

  "dense layer" should "produce right result (forward pass)" in {
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f),
      Array(1f, 0f, 1f),
      Array(1f, 1f, 1f))
    val weights = Tensor.matrix(
      Array(0f, 1f, 0.1f, 1f),
      Array(0f, 0.5f, 1f, 0f),
      Array(0f, 1f, 1f, 0.2f),
      Array(0f, 0.1f, 1f, 0.3f))
    val yExpected = Tensor.matrix(
      Array(0.7310586f, 0.5f, 0.549834f, 0.5744425f),
      Array(0.7502601f, 0.7310586f, 0.76852477f, 0.7858349f),
      Array(0.8807971f, 0.6224593f, 0.76852477f, 0.59868765f),
      Array(0.89090323f, 0.81757444f, 0.9002496f, 0.80218387f)
    )
    val model = Dense(4, Sigmoid)
    val result = model.result[Float].compile()
    result(x, Seq(weights)) should be(yExpected)
  }
}
