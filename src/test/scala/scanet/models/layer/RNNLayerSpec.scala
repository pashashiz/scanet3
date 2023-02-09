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

    "have string repr" in {
      val model = RNN(SimpleRNNCell(units = 2))
      model.toString shouldBe "RNN(SimpleRNNCell(2,GlorotUniform,Orthogonal,Zero,Zero) >> Bias(2,Zero,Zeros) >> Tanh,false)"
    }
  }

  "LSTM RNN layer" should {

    "calculate forward pass" in {
      val layer = RNN(LSTMCell(units = 2))
      val input = Tensor(Array(1f, 2f, 3f), Shape(1, 3, 1))
      val wf = Seq(
        Tensor.matrix(Array(0.17034012f, -0.39808878f)),
        Tensor.matrix(Array(0.4032409f, -0.37492862f), Array(-0.0409357f, 0.22285524f)),
        Tensor.vector(1f, 1f))
      val wi = Seq(
        Tensor.matrix(Array(0.5514972f, -0.6295431f)),
        Tensor.matrix(Array(-0.50807524f, -0.07556351f), Array(0.02593641f, 0.44693834f)),
        Tensor.vector(0f, 0f))
      val wg = Seq(
        Tensor.matrix(Array(0.39550006f, 0.03499722f)),
        Tensor.matrix(Array(-0.11996187f, 0.5238996f), Array(0.34965965f, -0.09114401f)),
        Tensor.vector(0f, 0f))
      val wo = Seq(
        Tensor.matrix(Array(-0.02363521f, 0.1830315f)),
        Tensor.matrix(Array(-0.37248448f, 0.07327155f), Array(-0.70414686f, -0.3490578f)),
        Tensor.vector(0f, 0f))
      val w = wf ++ wi ++ wg ++ wo
      val result = layer.result[Float].compile
      val prediction = result(input, w).const.roundAt(6).eval
      prediction shouldBe Tensor.matrix(Array(0.382158f, 0.029766f))
    }

    "have string repr" in {
      val model = RNN(LSTMCell(units = 2))
      model.toString shouldBe "RNN(LSTMCell(2,Tanh,Sigmoid,true,GlorotUniform,Orthogonal,Zeros,Ones,Zero,Zero,Zero),false)"
    }
  }
}
