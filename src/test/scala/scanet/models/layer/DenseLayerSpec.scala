package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Params.Weights
import scanet.core.{Params, Tensor}
import scanet.core.Path._
import scanet.models.Activation._
import scanet.models.Loss._
import scanet.models.Regularization.L2
import scanet.syntax._
import scanet.test.CustomMatchers

class DenseLayerSpec extends AnyWordSpec with CustomMatchers {

  "dense layer" should {

    "calculate forward pass" in {
      // input = (4 samples, 3 features)
      val x = Tensor.matrix(
        Array(0f, 0f, 1f),
        Array(0f, 1f, 1f),
        Array(1f, 0f, 1f),
        Array(1f, 1f, 1f))
      // weights = (3 features, 4 neurons)
      val w = Tensor.matrix(
        Array(1f, 0.5f, 1f, 0.1f),
        Array(0.1f, 1f, 1f, 1f),
        Array(1f, 0f, 0.2f, 0.3f))
      // bias = (4 neurons)
      val b = Tensor.vector(0f, 0f, 0f, 0f)
      val yExpected = Tensor.matrix(
        Array(0.731059f, 0.500000f, 0.549834f, 0.574442f),
        Array(0.750260f, 0.731059f, 0.768525f, 0.785835f),
        Array(0.880797f, 0.622459f, 0.768525f, 0.598688f),
        Array(0.890903f, 0.817574f, 0.900250f, 0.802184f))
      val model = Dense(4, Sigmoid)
      val forward = model.result[Float].compile
      val params = Params(0 / Weights -> w, 1 / Weights -> b)
      val y = forward(x, params).const.roundAt(6).eval
      y should be(yExpected)
    }

    "calculate penalty pass with regularization" in {
      val w = Tensor.matrix(
        Array(0f, 0f),
        Array(1f, 0.5f),
        Array(0.1f, 1f),
        Array(1f, 0f))
      val b = Tensor.vector(0f, 0f)
      val model = Dense(4, Sigmoid, reg = L2(lambda = 1))
      val params = Params(0 / Weights -> w, 1 / Weights -> b)
      model.penalty(params.mapValues(_.const)).eval should be(Tensor.scalar(1.63f))
    }

    "produce gradient when combined with loss function" in {
      val loss = Dense(4, Sigmoid).withLoss(BinaryCrossentropy)
      val grad = loss.grad[Float].compile
      val x = Tensor.matrix(
        Array(0f, 0f, 1f),
        Array(0f, 1f, 1f),
        Array(1f, 0f, 1f),
        Array(1f, 1f, 1f))
      val y = Tensor.matrix(
        Array(0.7310586f, 0.5f, 0.549834f, 0.5744425f),
        Array(0.7502601f, 0.7310586f, 0.76852477f, 0.7858349f),
        Array(0.8807971f, 0.6224593f, 0.76852477f, 0.59868765f),
        Array(0.89090323f, 0.81757444f, 0.9002496f, 0.80218387f))
      val weights = Tensor.zeros[Float](3, 4)
      val bias = Tensor.zeros[Float](4)
      val weightsGrad = Tensor.matrix(
        Array(-0.048231270f, -0.027502108f, -0.041798398f, -0.025054470f),
        Array(-0.040072710f, -0.034289565f, -0.041798398f, -0.036751173f),
        Array(-0.078313690f, -0.041943270f, -0.061695820f, -0.047571808f))
      val biasGrad = Tensor.vector(-0.078313690f, -0.041943270f, -0.061695820f, -0.047571808f)
      val before = Params(0 / Weights -> weights, 1 / Weights -> bias)
      val after = Params(0 / Weights -> weightsGrad, 1 / Weights -> biasGrad)
      grad(x, y, before) should be(after)
    }
  }
}
