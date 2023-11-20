package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.Params.Weights
import scanet.core.{Params, Tensor}
import scanet.core.Path._
import scanet.models.layer.BatchNorm._
import scanet.syntax._
import scanet.test.CustomMatchers

class BatchNormSpec extends AnyWordSpec with CustomMatchers {

  "batch norm layer" should {

    val input = Tensor.matrix(
      Array(0f, 0f, 10f),
      Array(0f, 8f, 4f),
      Array(5f, 0f, 13f),
      Array(20f, 3f, 8f))

    val baseParams = Params(
      "0" / Weights -> Tensor.matrix(
        Array(1f, 0.5f, 1f, 0.1f),
        Array(0.1f, 1f, 1f, 1f),
        Array(1f, 0f, 0.2f, 0.3f)),
      "1" / Weights -> Tensor.vector(0f, 0f, 0f, 0f))

    val model = Dense(4) >> BatchNorm(momentum = 0.5f)

    "normalize output to have 0 mean and 1 variance in training mode" in {
      val batchParams = Params(
        "2" / Beta -> Tensor.fill(1, 4)(0f),
        "2" / Gamma -> Tensor.fill(1, 4)(1f),
        "2" / MovingMean -> Tensor.fill(1, 4)(0f),
        "2" / MovingVariance -> Tensor.fill(1, 4)(1f))
      val forward = model.resultStateful[Float].compile
      val (outExpr, stateExpr) = forward(input, baseParams ++ batchParams)
      val out = outExpr.const.roundAt(3).eval
      val state = stateExpr.mapValues(_.const.roundAt(3)).eval
      out shouldBe Tensor.matrix(
        Array(-0.595f, -1.168f, -1.042f, -1.231f),
        Array(-1.181f, 0.422f, -0.232f, 1.313f),
        Array(0.307f, -0.671f, -0.375f, -0.656f),
        Array(1.469f, 1.417f, 1.649f, 0.574f))
      state("2" / MovingMean) shouldBe Tensor.matrix(
        Array(7.638f, 2.938f, 5.375f, 3.0f))
      state("2" / MovingVariance) shouldBe Tensor.matrix(
        Array(39.828f, 13.148f, 35.764f, 3.47f))
    }

    "use computed mean and variance in prediction mode" in {
      val batchParams = Params(
        "2" / Beta -> Tensor.fill(1, 4)(0f),
        "2" / Gamma -> Tensor.fill(1, 4)(1f),
        "2" / MovingMean -> Tensor.fill(1, 4)(5f),
        "2" / MovingVariance -> Tensor.fill(1, 4)(20f))
      val forward = model.freeze.result[Float].compile
      val outExpr = forward(input, baseParams ++ batchParams)
      val out = outExpr.const.roundAt(3).eval
      out shouldBe Tensor.matrix(
        Array(1.118f, -1.118f, -0.671f, -0.447f),
        Array(-0.045f, 0.671f, 0.85f, 0.939f),
        Array(2.907f, -0.559f, 0.581f, -0.134f),
        Array(5.21f, 1.789f, 4.383f, 0.537f))
    }
  }
}
