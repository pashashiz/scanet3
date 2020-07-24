package org.scanet.models.nn

import org.scalatest.flatspec.AnyFlatSpec
import org.scanet.core.Tensor
import org.scanet.models.{BinaryCrossentropy, Sigmoid}
import org.scanet.syntax._
import org.scanet.test.CustomMatchers

class DenseLayerSpec extends AnyFlatSpec with CustomMatchers {

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

  it should "produce right gradient when combined with loss function" in {
    val loss = Dense(4, Sigmoid).withLoss(BinaryCrossentropy)
    val grad = loss.grad[Float].compile()
    val x = Tensor.matrix(
      Array(0f, 0f, 1f),
      Array(0f, 1f, 1f),
      Array(1f, 0f, 1f),
      Array(1f, 1f, 1f))
    val y = Tensor.matrix(
      Array(0.7310586f, 0.5f, 0.549834f, 0.5744425f),
      Array(0.7502601f, 0.7310586f, 0.76852477f, 0.7858349f),
      Array(0.8807971f, 0.6224593f, 0.76852477f, 0.59868765f),
      Array(0.89090323f, 0.81757444f, 0.9002496f, 0.80218387f)
    )
    val weights = Tensor.zeros[Float](4, 4)
    val gradExpected = Tensor.matrix(
      Array(-0.078313690f, -0.048231270f, -0.040072710f, -0.078313690f),
      Array(-0.041943270f, -0.027502108f, -0.034289565f, -0.041943270f),
      Array(-0.061695820f, -0.041798398f, -0.041798398f, -0.061695820f),
      Array(-0.047571808f, -0.025054470f, -0.036751173f, -0.047571808f)
    )
    grad(x, y, Seq(weights)) should be(Seq(gradExpected))
  }
}
