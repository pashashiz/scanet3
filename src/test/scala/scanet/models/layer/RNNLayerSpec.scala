package scanet.models.layer

import org.scalatest.wordspec.AnyWordSpec
import scanet.core.{Shape, Tensor}
import scanet.models.Activation.Identity
import scanet.syntax._
import scanet.test.CustomMatchers

import scala.collection.immutable.Seq

class RNNLayerSpec extends AnyWordSpec with CustomMatchers {

  "Simple RNN layer" should {

    "calculate forward pass" in {
      val layer = RNN(SimpleRNNCell(units = 2, activation = Identity))
      val input = Tensor(Array(1f, 2f, 3f), Shape(1, 3, 1))
      val wx = Tensor.matrix(
        Array(-0.5302748f, -1.2049599f))
      val wh = Tensor.matrix(
        Array(-0.77547526f, -0.6313778f),
        Array(-0.6313778f, 0.7754754f))
      val b = Tensor.vector(0f, 0f)
      val expected = Tensor.matrix(Array(0.222901f, -6.019066f))
      val result = layer.result[Float].compile
      val prediction = result(input, Seq(wx, wh, b)).const.roundAt(6).eval
      prediction shouldBe expected
    }

    "have string repr" in {}
  }

  "LSTM RNN layer" should {

    "calculate forward pass" in {
      val layer = RNN(LSTMCell(units = 2))
      val input = Tensor(Array(1f, 2f, 3f), Shape(1, 3, 1))
      val wx = Tensor.matrix(
        Array(-0.5302748f, -1.2049599f))
      val wh = Tensor.matrix(
        Array(-0.77547526f, -0.6313778f),
        Array(-0.6313778f, 0.7754754f))
      val b = Tensor.vector(0f, 0f)
      val w = Seq.fill(4)(Seq(wx, wh, b)).flatten
      val result = layer.result[Float].compile
      val prediction = result(input, w).const.roundAt(6).eval
      println(prediction)
      // todo: match expected
    }

    "have string repr" in {}
  }
}
